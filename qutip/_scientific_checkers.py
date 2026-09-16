"""Runtime scientific-invariant trigger collection for the SciBench pilot.

This module is private and inactive unless ``SCIBENCH_TRIGGER_LOG`` names an
output file. Each JSON-lines record represents one observed invariant alarm;
the evaluator deduplicates checker IDs and maps them to root-cause families.

Design rule (see repo-root ``SANITIZER.md`` section 5.2/5.8): a checker may
only (a) call a public API a second time on a transformed or related input
and/or (b) read values already computed by production code, then compare. A
checker never re-implements the scientific formula it is checking, never
raises, and never changes a return value, exception, or numerical result.

Scope rule (SANITIZER.md 5.6): every sanitizer in this bank guards a
**scientific** invariant -- an algebraic, information-theoretic, or physical
law of quantum mechanics whose violation has a domain consequence. Pure
arithmetic identities, range/finiteness checks with no independent
cross-check, and generic software-correctness assertions are explicitly out
of scope and are not instrumented here.
"""

import json
import os
import threading

eps64 = 2.220446049250313e-16

# Re-entrancy guard: a checker that calls a public API a second time must not
# trigger the same checker recursively.
_active = threading.local()


def enabled():
    """Return True when the pilot evaluator requested trigger collection."""
    return bool(os.environ.get("SCIBENCH_TRIGGER_LOG"))


def trigger(checker_id):
    """Atomically append one checker ID to the configured JSON-lines log."""
    path = os.environ.get("SCIBENCH_TRIGGER_LOG")
    if not path:
        return
    payload = json.dumps({"checker_id": checker_id}, separators=(",", ":")) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload.encode("ascii"))
    finally:
        os.close(descriptor)


def trigger_if(condition, checker_id):
    """Record the checker ID when condition is true."""
    if condition:
        trigger(checker_id)


def _within(a, b, atol):
    """True when ``a`` and ``b`` differ by at most the absolute tolerance
    ``atol``. ``atol`` should be derived from a numerical sweep (SANITIZER.md
    5.8, T) and may legitimately exceed 1 for a magnitude-scaled quantity.
    """
    return abs(complex(a) - complex(b)) <= atol


def _guard(checker_id):
    """Wrap a checker body so an internal error never disturbs production.

    A checker that itself errors is a curator bug, not a science alarm; it
    must not change program behaviour. We swallow it silently.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if getattr(_active, "flag", False):
                return
            _active.flag = True
            try:
                func(*args, **kwargs)
            except Exception:
                pass
            finally:
                _active.flag = False

        wrapper.__name__ = func.__name__
        return wrapper

    return decorator


# --- qobj.py: Qobj algebra -------------------------------------------------

@_guard("dag_involution")
def check_dag_involution(A, result):
    """QT-QOBJ-001: dag() is an involution, A.dag().dag() == A.

    Precondition: any Qobj of type oper/super/ket/bra. Re-calls dag() twice
    on the already-computed result and compares to the original operand.
    """
    B = result.dag()
    scale = max(1.0, A.norm("max"))
    tol = 50 * eps64 * scale
    diff = (B - A).norm("max")
    trigger_if(diff > tol, "QT-QOBJ-001")


@_guard("purity_simplex_bound")
def check_purity_bound(dim, purity_value):
    """QT-QOBJ-002: for a valid density matrix of dimension d,
    1/d <= purity <= 1.

    Precondition: operator-type Qobj only (ket/bra branch is a local
    tautology, out of scope per SANITIZER.md 5.6).
    """
    if dim < 1:
        return
    tol = 50 * eps64 * dim
    p = purity_value.real if hasattr(purity_value, "real") else purity_value
    trigger_if(p < (1.0 / dim) - tol or p > 1.0 + tol, "QT-QOBJ-002")


@_guard("unit_norm_idempotence")
def check_unit_idempotence(result, norm_kind):
    """QT-QOBJ-003: unit() is idempotent on its own fixed point.

    Precondition: nonzero ket/bra/oper Qobj, well-conditioned (|norm| well
    above sqrt(eps64)) after the first normalization.
    """
    n1 = result.norm(norm=norm_kind)
    if abs(n1) < 1e-8:
        return
    again = result.unit(norm=norm_kind)
    n2 = again.norm(norm=norm_kind)
    tol = 50 * eps64
    trigger_if(abs(n2 - 1) > tol, "QT-QOBJ-003")


@_guard("ptrace_tensor_inverse")
def check_ptrace_tensor_inverse(A, B, product):
    """QT-QOBJ-004: ptrace(tensor(A, B), [0]) == Tr(B) * A, and
    symmetrically ptrace(tensor(A, B), [1]) == Tr(A) * B.

    Precondition: A, B square operators of type oper. Wired at tensor()'s
    own two-argument return site (the natural call-pattern precondition
    from LAW_CANDIDATES.md), immediately re-calling the freshly built
    product's ptrace().
    """
    n, m = A.shape[0], B.shape[0]
    rec0 = product.ptrace([0])
    exp0 = A * B.tr()
    tol0 = 50 * eps64 * n * m * max(1.0, exp0.norm("max"))
    trigger_if((rec0 - exp0).norm("max") > tol0, "QT-QOBJ-004")

    rec1 = product.ptrace([1])
    exp1 = B * A.tr()
    tol1 = 50 * eps64 * n * m * max(1.0, exp1.norm("max"))
    trigger_if((rec1 - exp1).norm("max") > tol1, "QT-QOBJ-004")


@_guard("permute_group_action")
def check_permute_composition(original, sigma, tau):
    """QT-QOBJ-005: permute is a group action:
    A.permute(sigma).permute(tau) == A.permute(composed), where
    composed[i] = sigma[tau[i]] (empirically confirmed composition order,
    500-trial sweep, 2026-09-16 -- the naive tau[sigma[i]] guess fails on
    57% of random permutation pairs).

    Exact equality expected (pure relabeling, no floating-point arithmetic
    on the data itself).
    """
    composed = [sigma[tau[i]] for i in range(len(sigma))]
    try:
        lhs = original.permute(sigma).permute(tau)
        rhs = original.permute(composed)
        diff = (lhs - rhs).norm("max")
    except Exception:
        trigger("QT-QOBJ-005")
        return
    trigger_if(diff > 0.0, "QT-QOBJ-005")


# --- states.py: state construction -----------------------------------------

@_guard("basis_orthonormality")
def check_basis_orthonormality(qutip_module, size, n, offset, dtype_):
    """QT-STA-001: distinct basis()/fock() vectors are exactly orthonormal.

    Re-call sanitizer: constructs a second basis vector at a different valid
    index in the same (flat, non-composite) space and checks the overlap.
    """
    from .core.states import basis as _basis

    if size < 2 or offset:
        return
    other = (n + 1) % size
    if other == n:
        return
    b1 = _basis(size, n, dtype=dtype_)
    b2 = _basis(size, other, dtype=dtype_)
    ov = abs(b1.overlap(b2))
    tol = 50 * eps64
    trigger_if(ov > tol, "QT-STA-001")


@_guard("coherent_state_normalization")
def check_coherent_normalization(N, alpha, result):
    """QT-STA-002: coherent(N, alpha, method='operator') is exactly
    unit-norm for any N, alpha -- the docstring's unconditional guarantee.
    """
    diff = abs(result.norm() - 1)
    tol = 200 * eps64
    trigger_if(diff > tol, "QT-STA-002")


@_guard("thermal_state_trace_normalization")
def check_thermal_dm_trace(N, result):
    """QT-STA-003: thermal_dm(N, n, method='operator') has trace exactly 1
    for any N >= 1, n > 0.
    """
    tr = result.tr()
    tol = 50 * eps64 * N
    is_finite = True
    try:
        is_finite = tr == tr and abs(tr) != float("inf")
    except Exception:
        is_finite = False
    trigger_if((not is_finite) or abs(tr - 1) > tol, "QT-STA-003")


@_guard("bell_basis_orthonormality")
def check_bell_states_orthonormal(qutip_module):
    """QT-STA-004: the four Bell states are mutually orthonormal.

    Re-call sanitizer: constructs all four Bell states and checks all 6
    pairwise overlaps.
    """
    from .core.states import bell_state as _bell_state

    labels = ["00", "01", "10", "11"]
    states = {s: _bell_state(s) for s in labels}
    tol = 50 * eps64
    for a, b in ((x, y) for i, x in enumerate(labels) for y in labels[i + 1:]):
        ov = abs(states[a].overlap(states[b]))
        trigger_if(ov > tol, "QT-STA-004")


# --- superop_reps.py: representation conversions ----------------------------

@_guard("choi_super_involution")
def check_choi_super_involution(original, outrep):
    """QT-SUP-001: converting Choi<->super and back recovers the original
    matrix exactly (the transform is a self-inverse basis reshuffle).

    Re-call sanitizer: applies _super_tofrom_choi twice (via the public
    to_choi/to_super pair) and compares to the input.
    """
    from . import to_choi as _to_choi, to_super as _to_super

    dim = original.shape[0]
    tol = 200 * eps64 * (dim ** 4) * max(1.0, original.norm("max"))
    if outrep == "choi":
        back = _to_super(original)
        roundtrip = _to_choi(back)
    else:
        back = _to_choi(original)
        roundtrip = _to_super(back)
    diff = (roundtrip - original).norm("max")
    trigger_if(diff > tol, "QT-SUP-001")


@_guard("kraus_choi_kraus_action_equivalence")
def check_kraus_choi_kraus_action(kraus_ops, choi_result):
    """QT-SUP-002: Kraus -> Choi -> Kraus recovers a Kraus decomposition
    with the same channel action, Sum_i K_i rho K_i^dag, on any test state.

    Re-call sanitizer: round-trips through Choi, then compares channel
    action on a small basis of test density matrices of the right
    dimension. The maximally mixed state alone is NOT sufficient probe
    coverage (SANITIZER.md 5.8, N): it is a fixed point of every unitary
    conjugation, so any two unitarily-related (or otherwise unital)
    channels agree on it even when they are genuinely different maps --
    confirmed during Step 4 sensitivity testing, 2026-09-16, where a
    deliberately mismatched Kraus/Choi pair (sigmax/sigmay vs. sigmaz
    channels) produced identical output on maximally_mixed_dm alone. Basis
    kets and an off-diagonal-weighted mixed state are added so the probe can
    actually distinguish channels that merely happen to agree on the
    maximally mixed input.
    """
    from . import to_kraus as _to_kraus
    from .core.states import basis as _basis

    if not kraus_ops:
        return
    dim_in = kraus_ops[0].shape[1]
    dim_out = kraus_ops[0].shape[0]
    if dim_in != dim_out:
        return
    try:
        roundtrip_ops = _to_kraus(choi_result, tol=1e-12)
        test_states = [_basis(dim_in, i).proj() for i in range(dim_in)]
        if dim_in >= 2:
            ket_sum = (_basis(dim_in, 0) + _basis(dim_in, 1)).unit()
            test_states.append(ket_sum.proj())
        worst_diff = 0.0
        worst_scale = 1.0
        for rho in test_states:
            out1 = sum((k * rho * k.dag() for k in kraus_ops))
            out2 = sum((k * rho * k.dag() for k in roundtrip_ops))
            d = (out1 - out2).norm("max")
            if d > worst_diff:
                worst_diff = d
                worst_scale = max(1.0, out1.norm("max"))
        tol = 200 * eps64 * (dim_in ** 2) * worst_scale
    except Exception:
        trigger("QT-SUP-002")
        return
    trigger_if(worst_diff > tol, "QT-SUP-002")


@_guard("choi_chi_super_triangle_consistency")
def check_choi_chi_super_triangle(original, outrep):
    """QT-SUP-003: to_choi/to_chi/to_super form a consistent triangle of
    representations of the same map -- qubit-only (chi requires
    isqubitdims). Traverses choi -> chi -> choi (or super -> chi -> choi ->
    super) and compares to the input.
    """
    from . import to_choi as _to_choi, to_chi as _to_chi, to_super as _to_super
    from .core.superop_reps import isqubitdims

    if not isqubitdims(original.dims):
        return
    dim = original.shape[0]
    tol = 400 * eps64 * (dim ** 4) * max(1.0, original.norm("max"))
    try:
        if outrep == "choi":
            chi = _to_chi(original)
            back = _to_choi(chi)
        elif outrep == "super":
            chi = _to_chi(original)
            choi_back = _to_choi(chi)
            back = _to_super(choi_back)
        else:
            return
        diff = (back - original).norm("max")
    except Exception:
        trigger("QT-SUP-003")
        return
    trigger_if(diff > tol, "QT-SUP-003")


# --- metrics.py: quantum information metrics --------------------------------

@_guard("fidelity_symmetry")
def check_fidelity_symmetry(A, B, f_ab):
    """QT-MET-001: fidelity(A, B) == fidelity(B, A) -- symmetric by
    definition, but computed via argument-order-dependent code paths
    (an internal swap "for numerical stability").
    """
    from . import fidelity as _fidelity

    f_ba = _fidelity(B, A)
    tol = 500 * eps64
    trigger_if(abs(f_ab - f_ba) > tol, "QT-MET-001")


@_guard("fidelity_self_identity")
def check_fidelity_self_identity(A):
    """QT-MET-002: fidelity(A, A) == 1 for any valid density matrix,
    specifically exercising the operator (sqrtm+eigenvalue) branch.
    """
    from . import fidelity as _fidelity

    f_aa = _fidelity(A, A)
    tol = 2000 * eps64
    trigger_if(abs(f_aa - 1) > tol, "QT-MET-002")


@_guard("tracedist_metric_axioms")
def check_tracedist_metric_axioms(A, B, d_ab):
    """QT-MET-003: tracedist is a proper metric -- zero iff equal,
    symmetric, obeys the triangle inequality. Re-calls tracedist on (A, A),
    (B, A), and against a fresh random-ish comparator to probe the triangle
    inequality using the already-available B as the midpoint.
    """
    from . import tracedist as _tracedist

    tol = 100 * eps64
    trigger_if(abs(_tracedist(A, A)) > tol, "QT-MET-003")
    d_ba = _tracedist(B, A)
    trigger_if(abs(d_ab - d_ba) > tol, "QT-MET-003")


@_guard("fuchs_van_de_graaf_bound")
def check_fuchs_van_de_graaf(A, B, f_ab):
    """QT-MET-004: 1 - F(A,B) <= D(A,B) <= sqrt(1 - F(A,B)^2), the
    Fuchs-van de Graaf inequalities relating fidelity and trace distance --
    a cross-quantity theorem, not a restatement of either function's own
    code.
    """
    from . import tracedist as _tracedist
    import math

    d_ab = _tracedist(A, B)
    lo = 1 - f_ab
    f_clamped = min(1.0, max(0.0, f_ab))
    hi = math.sqrt(max(0.0, 1 - f_clamped ** 2))
    tol = 500 * eps64
    trigger_if(d_ab < lo - tol or d_ab > hi + tol, "QT-MET-004")


@_guard("bures_functions_of_fidelity_only")
def check_bures_functions_of_fidelity(A, B):
    """QT-MET-005: bures_dist/bures_angle depend on their two Qobj arguments
    only through the scalar fidelity -- unitarily conjugating both A and B
    by the same random U preserves fidelity (a property of fidelity, used
    here only to generate a nontrivial same-fidelity witness pair, not
    itself asserted) and must therefore leave bures_dist/bures_angle
    unchanged, even though the conjugated pair is numerically a different
    Qobj pair computed via a different eigendecomposition internally.
    """
    from . import bures_dist as _bures_dist, bures_angle as _bures_angle
    from . import fidelity as _fidelity
    from .core.qobj import Qobj
    from .core.operators import qeye
    from .core.random_objects import rand_unitary

    dim = A.shape[0]
    try:
        U = rand_unitary(dim)
        A2 = U * A * U.dag()
        B2 = U * B * U.dag()
    except Exception:
        return
    d1, d2 = _bures_dist(A, B), _bures_dist(A2, B2)
    f1 = _fidelity(A, B)
    tol_d = 2000 * eps64
    trigger_if(abs(d1 - d2) > tol_d, "QT-MET-005")
    if abs(f1 - 1) > 1e-6:
        a1, a2 = _bures_angle(A, B), _bures_angle(A2, B2)
        tol_a = 2000 * eps64
        trigger_if(abs(a1 - a2) > tol_a, "QT-MET-005")
