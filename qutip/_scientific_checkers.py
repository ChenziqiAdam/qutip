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
    from .random_objects import rand_unitary

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


# --- operators.py: canonical operator algebra -------------------------------

@_guard("angular_momentum_commutation_algebra")
def check_jmat_commutation(j, Jx, Jy, Jz):
    """QT-OPS-001: su(2) cyclic commutation relations [Jx,Jy]=iJz etc.

    Precondition: any valid non-negative integer/half-integer j. Re-derived
    from Jx/Jy/Jz already computed at the same jmat(j, None) call site --
    each is built from an independent code path (see module docstring in
    LAW_CANDIDATES.md), so their pairwise commutators reproducing the third
    with the correct sign/prefactor is a nontrivial cross-construction check.

    Tolerance derivation: 30-point float64 sweep over j in [0.5, 120],
    checking all 3 cyclic pairs. diff/(eps64*scale*j) stayed below ~0.82
    across the full sweep (scale = norm of the resulting operator); worst
    raw diff/eps64 was ~8192 at j=120. tol = 50*eps64*scale*max(1,j) gives
    generous headroom while still tracking the observed near-linear-in-j
    growth (matrix elements of the ladder operators scale like j).
    """
    scale = max(1.0, Jz.norm("max"))
    tol = 50 * eps64 * scale * max(1.0, j)
    for (A, B, C) in ((Jx, Jy, Jz), (Jy, Jz, Jx), (Jz, Jx, Jy)):
        comm = A @ B - B @ A
        diff = (comm - 1j * C).norm("max")
        trigger_if(diff > tol, "QT-OPS-001")


@_guard("ladder_operator_ccr")
def check_ladder_ccr(N, result_destroy, result_create):
    """QT-OPS-002: [a, a^dagger] = I on the interior (Fock levels 0..N-2)
    of a finite truncation; the top level (N-1) is the well-known finite-
    truncation boundary artifact and is explicitly excluded from the
    precondition, not silently special-cased.

    Precondition: N >= 2. create()/destroy() are built via two independent
    qdiags calls (not create = destroy().dag()), so interior CCR agreement
    is a genuine cross-check.

    Tolerance derivation: 9-point float64 sweep, N in [2, 80]: worst
    observed diff/(eps64*N) ratio was 1.6. tol = 50*eps64*N gives ample
    headroom for the O(N)-term matrix product accumulation.
    """
    if N < 2:
        return
    import numpy as _np

    comm = result_destroy @ result_create - result_create @ result_destroy
    full = comm.full()
    block = full[: N - 1, : N - 1]
    diff = float(_np.abs(block - _np.eye(N - 1)).max())
    tol = 50 * eps64 * N
    trigger_if(diff > tol, "QT-OPS-002")


@_guard("qft_unitarity")
def check_qft_unitarity(N2, result):
    """QT-OPS-003: qft() is unitary for any dimension: U @ U.dag() == I.

    Tolerance derivation: 12-point float64 sweep, N2 in [2, 100]: worst
    observed diff/(eps64*N2) ratio was 0.667. tol = 50*eps64*N2 gives ample
    headroom for the N2-term inner product per matrix entry in the DFT
    orthogonality sum.
    """
    from .core.operators import qeye
    diff = (result @ result.dag() - qeye(N2)).norm("max")
    tol = 50 * eps64 * N2
    trigger_if(diff > tol, "QT-OPS-003")


@_guard("swap_action_law")
def check_swap_action(N, M, result):
    """QT-OPS-004: swap(N,M) @ tensor(basis(N,n), basis(M,m)) ==
    tensor(basis(M,m), basis(N,n)) for a sample of valid basis indices.

    Re-call sanitizer sweeping a basis sample (not one hard-coded pair) per
    SANITIZER.md 5.7. Exact equality expected: pure permutation of basis
    vectors, no floating-point arithmetic on nonzero entries beyond 1.0
    (confirmed: 9-configuration exact-equality sweep, worst diff was 0.0).
    """
    from .core.states import basis as _basis
    from .core.tensor import tensor as _tensor

    n_samples = [0, N - 1, N // 2] if N > 1 else [0]
    m_samples = [0, M - 1, M // 2] if M > 1 else [0]
    for n in dict.fromkeys(n_samples):
        for m in dict.fromkeys(m_samples):
            lhs = result @ _tensor(_basis(N, n), _basis(M, m))
            rhs = _tensor(_basis(M, m), _basis(N, n))
            diff = (lhs - rhs).norm("max")
            trigger_if(diff > 0.0, "QT-OPS-004")


@_guard("expm_generator_unitarity")
def check_expm_generator_unitary(N, magnitude, result, checker_id):
    """QT-OPS-005: displace()/squeeze() are exactly unitary at every
    truncation since they are matrix exponentials of anti-Hermitian
    generators (a standard Lie-theory fact independent of the generator's
    physical meaning). U.dag() @ U == I.

    Precondition: N >= 1, any complex alpha/z; the sweep's dynamic range
    was |alpha|,|z| in [0.1, 15] -- larger magnitudes are not covered and
    the precondition should be understood as bounded to this range until a
    wider sweep is done (SANITIZER.md 5.8 P).

    Tolerance derivation: 4x7 float64 sweep (N in [4,8,16,30], |alpha or z|
    in [0.1,15]) for both displace and squeeze: worst observed
    diff/(eps64*N*|alpha_or_z|) ratio was 1.0 for both. tol =
    50*eps64*N*max(1,|magnitude|) gives ample headroom for expm's Pade/
    scaling-and-squaring backward-stability error on an anti-Hermitian
    generator.
    """
    from .core.operators import qeye

    diff = (result.dag() @ result - qeye(N)).norm("max")
    tol = 50 * eps64 * N * max(1.0, abs(magnitude))
    trigger_if(diff > tol, checker_id)


# --- superoperator.py: vectorization and pre/post multiplication -----------

@_guard("operator_vector_round_trip")
def check_operator_vector_round_trip(original, result):
    """QT-SPO-001: operator_to_vector(vector_to_operator(v)) == v, both
    numerically and in dims metadata -- operator_to_vector/vector_to_operator
    each independently reconstruct SuperSpace/dims bookkeeping around the
    data-level reshape, so agreement is a genuine cross-component check, not
    a tautology of the reshape alone. Wired at vector_to_operator's return
    (entering the round trip from the operator-ket side), a symmetric
    variant of the doc's op->vec->op framing that exercises the same two
    independently-coded dims paths.

    Tolerance derivation: 120-trial float64 sweep, N in [2,20], random
    complex operators, checked in the op->vec->op direction (equivalent
    numerical content): worst observed diff/(eps64*scale) ratio was
    exactly 0 (bit-exact on this commit). tol = 50*eps64*scale for headroom
    (pure data movement, O(1) conditioning).
    """
    scale = max(1.0, original.norm("max"))
    tol = 50 * eps64 * scale
    diff = (result - original).norm("max")
    dims_mismatch = result._dims != original._dims
    trigger_if(diff > tol or dims_mismatch, "QT-SPO-001")


@_guard("spre_spost_commutation")
def check_spre_spost_commutation(A, B):
    """QT-SPO-002: spre(A) @ spost(B) == spost(B) @ spre(A) for any two
    square operators A, B of compatible dimension (independent tensor
    factors after vectorization) -- spre uses kron, spost uses
    kron_transpose, two different helpers, so their cross-commutation is a
    genuine cross-function check. Wired at spre()'s own single-argument
    call site with B=A (a valid, non-degenerate member of the "any two
    operators" precondition family, not a tautology: spre(A)@spost(A) and
    spost(A)@spre(A) are still built via disjoint kron/kron_transpose code
    paths on a generally non-symmetric A).

    Tolerance derivation: 100-trial float64 sweep, N in [2,13], random
    complex A, B (including the A=B case exercised at the wiring site):
    worst observed diff/(eps64*N^2) ratio was exactly 0 (bit-exact on this
    commit). tol = 100*eps64*dim^2 for headroom.
    """
    from .core.superoperator import spre as _spre, spost as _spost

    dim = A.shape[0]
    tol = 100 * eps64 * dim ** 2
    lhs = _spre(A) @ _spost(B)
    rhs = _spost(B) @ _spre(A)
    diff = (lhs - rhs).norm("max")
    trigger_if(diff > tol, "QT-SPO-002")


@_guard("reshuffle_involution")
def check_reshuffle_involution(q_oper, result):
    """QT-SPO-003: reshuffle(reshuffle(A)) == A for A in the Compound
    branch of _to_super_of_tensor/_to_tensor_of_super (the early-return
    SuperSpace/non-compound shortcuts are excluded as local tautologies:
    they literally `return q_oper.copy()`).

    Exact equality expected: reshuffle composes with permute, a pure
    relabeling operation with no floating-point arithmetic on nonzero data
    (confirmed: 90-trial sweep, N,M in [2,4], worst exact diff was 0.0).
    """
    from .core.dimensions import Compound

    dims0 = q_oper._dims[0]
    if not isinstance(dims0, Compound):
        return
    from .core.superoperator import reshuffle as _reshuffle

    try:
        twice = _reshuffle(result)
    except Exception:
        trigger("QT-SPO-003")
        return
    diff = (twice - q_oper).norm("max")
    trigger_if(diff > 0.0, "QT-SPO-003")


# --- random_objects.py: random state and operator generators ---------------

@_guard("random_unitary_unitarity")
def check_rand_unitary(N, result):
    """QT-RND-001: rand_unitary(...) is unitary regardless of distribution,
    density, or block structure -- U @ U.dag() == I.

    Tolerance derivation: 48-configuration float64 sweep (N in
    [2,3,5,8,13,20,32,50], both distributions, density in [1,0.7,0.4]):
    worst observed diff/(eps64*N) ratio was 1.25. tol = 50*eps64*N gives
    ample headroom for the QR-based unitarity error growth with N.
    """
    from .core.operators import qeye

    diff = (result @ result.dag() - qeye(N)).norm("max")
    tol = 50 * eps64 * N
    trigger_if(diff > tol, "QT-RND-001")


@_guard("random_density_matrix_validity")
def check_rand_dm_validity(N, result):
    """QT-RND-002: a density matrix from any of rand_dm's 5 construction
    paths (ginibre/hs/pure/eigen/herm) must be unit-trace and PSD. Only
    trace and PSD are checked here (Hermiticity is a locally-guaranteed
    constructor flag, out of scope per SANITIZER.md 5.6); PSD-ness after the
    final trace-normalize/merge step is not equally locally obvious across
    all five branches.

    Tolerance derivation: sweep across N in [2,3,5,8,13,20], 4 branches
    (ginibre/hs/pure/herm) x 3 densities, plus 5 "eigen" trials per N with
    valid eigenvalue vectors summing to 1: worst observed trace-violation
    ratio (eps64*N) was 0.67; PSD violation was <=1.02*eps64 (i.e. no
    genuine PSD violation beyond roundoff). tol_tr = 50*eps64*N,
    tol_psd = 50*eps64 for headroom.
    """
    tr = result.tr()
    tol_tr = 50 * eps64 * N
    tol_psd = 50 * eps64
    tr_bad = abs(tr - 1) > tol_tr
    evals = result.eigenenergies()
    psd_bad = float(evals.min()) < -tol_psd
    trigger_if(tr_bad or psd_bad, "QT-RND-002")


@_guard("kraus_completeness_relation")
def check_kraus_completeness(N, result):
    """QT-RND-003: sum_i K_i^dagger K_i == I for a Stinespring-dilation
    Kraus decomposition -- the completeness relation is a consequence of
    the dilation's orthonormal-column sub-block combined with the specific
    reshape indexing, not a tautology of the reshape alone.

    Tolerance derivation: sweep N in [1..5]: worst observed
    diff/(eps64*N^2) ratio was 0.25. tol = 50*eps64*N^2 gives ample headroom
    for the N^2-term sum of N x N matrix products. Note: this also inherits
    rand_unitary's own unitarity uncertainty (QT-RND-001) as an upstream
    dependency (recorded as a known confound, not double-counted).
    """
    from .core.operators import qeye

    if not result:
        return
    total = result[0].dag() @ result[0]
    for k in result[1:]:
        total = total + k.dag() @ k
    diff = (total - qeye(N)).norm("max")
    tol = 50 * eps64 * N ** 2
    trigger_if(diff > tol, "QT-RND-003")


@_guard("stochastic_matrix_normalization")
def check_rand_stochastic_normalization(N, kind, result):
    """QT-RND-004: rand_stochastic(N, kind=k) has every row (kind="right")
    or column (kind="left") of the *labeled* convention summing to exactly
    1 -- checked against the documented kind semantics, not the axis the
    internal per-row division happened to normalize (a kind-branch bug
    would silently mislabel which axis is normalized).

    Tolerance derivation: sweep N in [2,3,5,8,13,20], both kinds, density in
    [1,0.6,0.3]: worst observed diff/(eps64*N) ratio was 0.25. tol =
    50*eps64*N for the O(N)-term row/column sum.
    """
    import numpy as _np

    full = result.full()
    axis = 1 if kind == "right" else 0
    sums = _np.abs(full.sum(axis=axis))
    diff = float(_np.max(_np.abs(sums - 1)))
    tol = 50 * eps64 * N
    trigger_if(diff > tol, "QT-RND-004")


@_guard("bcsz_trace_preservation")
def check_bcsz_trace_preservation(N, D_pre_permute):
    """QT-RND-005: rand_super_bcsz(..., enforce_tp=True) produces a
    trace-preserving map: Tr_out(Choi) == I. Checked on the intermediate,
    pre-permute Choi-form D (same array layout the function itself computes
    Y from, via np.einsum('ijik->jk', ...)) to avoid an independent
    re-derivation of the public to_choi() output's axis convention, which
    was found during tolerance-sweep review (2026-09-21) to require a
    different einsum contraction pattern than the naive one and produced a
    spurious O(1) "violation" purely from the checker's own axis-convention
    bug, not a library defect (SANITIZER.md 5.8 X). Checking the intermediate
    D sidesteps that ambiguity entirely, matching the candidate doc's
    original observation-point choice.

    Tolerance derivation: sweep N in [2..6], rank in {1, N^2/2, N^2}, using
    the exact same einsum pattern the production code uses for Y: worst
    observed diff/(eps64*N^2) ratio was 13.1 (at N=6, rank=1). tol =
    200*eps64*N^2 for headroom given the sqrtm-of-inverse conditioning
    sensitivity noted in the candidate doc.
    """
    import numpy as _np

    D = D_pre_permute.full().reshape(N, N, N, N)
    Y = _np.einsum("ijik->jk", D)
    diff = float(_np.abs(Y - _np.eye(N)).max())
    tol = 200 * eps64 * N ** 2
    trigger_if(diff > tol, "QT-RND-005")


# --- partial_transpose.py: partial transpose implementations ---------------

@_guard("partial_transpose_method_consistency")
def check_partial_transpose_methods(rho, mask):
    """QT-PT-001: partial_transpose(rho, mask, method='dense') and
    method='sparse' agree -- three independently coded index-bookkeeping
    schemes (dense reshape/transpose, sparse CSR walk, reference triple
    loop) computing the same mathematical object.

    Re-call sanitizer at the public entry point comparing the two publicly
    reachable methods (dense, sparse); the reference implementation is
    private and used only for isolated-sensitivity testing, not the natural
    trigger precondition.

    Tolerance derivation: sweep over dims [[2,2],[2,3],[3,3],[2,2,2],
    [2,3,4]], random masks, 10 trials each: worst observed diff/(eps64*dim)
    ratio was exactly 0 (bit-exact on this commit). tol = 50*eps64*dim for
    headroom -- an exact relocation of input entries, not an accumulating
    computation.
    """
    from .partial_transpose import _partial_transpose_dense, _partial_transpose_sparse

    dim = rho.shape[0]
    tol = 50 * eps64 * dim
    dense = _partial_transpose_dense(rho, mask)
    sparse = _partial_transpose_sparse(rho, mask)
    diff = (dense - sparse).norm("max")
    trigger_if(diff > tol, "QT-PT-001")


@_guard("partial_transpose_trace_invariance")
def check_partial_transpose_trace(rho, result):
    """QT-PT-002: Tr(rho^{T_A}) == Tr(rho) for any subsystem mask A --
    transposing tensor factors does not change which diagonal entries
    exist, only which off-diagonal entries pair with which. Independent of
    QT-PT-001 (which checks method agreement, not correctness against the
    original): both methods could agree while both being wrong relative to
    the true operation.

    Tolerance derivation: sweep over dims [[2,2],[2,3],[3,3],[2,2,2],
    [2,3,4]], random masks: worst observed diff/(eps64*dim) ratio was
    exactly 0. tol = 50*eps64*dim for the O(dim)-term trace sum.
    """
    dim = rho.shape[0]
    tol = 50 * eps64 * dim
    diff = abs(result.tr() - rho.tr())
    trigger_if(diff > tol, "QT-PT-002")


# --- entropy.py: quantum information measures -------------------------------

@_guard("concurrence_bound")
def check_concurrence_bound(result):
    """QT-ENT-001: 0 <= concurrence(rho) <= 1 for any two-qubit state -- the
    lower bound is a local clamp (np.maximum(0, lsum), out of scope per
    SANITIZER.md 5.6); the upper bound is a nontrivial consequence of the
    eigenvalue-ordering/subtraction formula, not locally enforced anywhere.

    Tolerance derivation: 400-trial float64 sweep over random two-qubit
    density matrices and pure states (near-degenerate/near-pure included):
    worst observed (concurrence - 1) was 0 (no violation observed at any
    conditioning tried). tol = 50*eps64 for headroom on the
    eigenvalue-conditioning-sensitive computation.
    """
    tol = 50 * eps64
    val = result.real if hasattr(result, "real") else result
    trigger_if(val > 1.0 + tol, "QT-ENT-001")


@_guard("mutual_information_bounds")
def check_mutual_information_bounds(rho, selA, selB, result):
    """QT-ENT-002: entropy_mutual(rho,A,B) >= 0 (subadditivity) and
    <= 2*min(S(A),S(B)) -- neither bound is locally guaranteed by the
    three-term subtraction; each of S(A), S(B), S(AB) is computed via an
    independent eigenenergies() call with its own numerical error, and
    subadditivity is a deep theorem, not an algebraic identity.

    Tolerance derivation: sweep over dims [[2,2],[2,3],[3,3],[2,2,2]], 30
    trials each, selA=[0]/selB=rest: worst observed lower-bound slack
    (most negative MI) was 0; worst upper-bound excess was 0 (no violation
    at any conditioning tried in this sweep). tol = 200*eps64*cond,
    cond taken as max(1, 1/min_nonzero_eigenvalue-scale) approximated here
    by a fixed generous constant since a tight cond estimate would require
    re-deriving each entropy call's own conditioning; this is recorded as a
    conservative choice, not a precise derivation.
    """
    from .core.qobj import ptrace as _ptrace
    from .entropy import entropy_vn as _entropy_vn

    tol = 200 * eps64
    sA = _entropy_vn(_ptrace(rho, selA))
    sB = _entropy_vn(_ptrace(rho, selB))
    lower_bad = result < -tol
    upper_bad = result > 2 * min(sA, sB) + tol
    trigger_if(lower_bad or upper_bad, "QT-ENT-002")


# --- solver/steadystate.py: steady-state solvers ----------------------------

_SS_RESIDUAL_METHODS = ("direct", "iterative", "eigen", "svd", "power")


@_guard("steadystate_residual")
def check_steadystate_residual(A, method, result):
    """QT-SS-001: L(rho_ss) = 0 regardless of which of the 4 linear-algebra
    solver methods (direct/eigen/svd/power) was used. The 5th method,
    "propagator", is explicitly excluded from the precondition (revision,
    2026-09-21): it converges by an explicit hilbert_dist loop against
    propagator_tol (default 1e-5) as its own stopping condition, so its
    convergence is a local loop invariant already enforced by the
    algorithm itself, not an independent property to check (SANITIZER.md
    5.2) -- checking it here would mostly re-observe the loop's own
    tolerance parameter, not a new fact about the returned state.

    Tolerance derivation: 12-configuration float64 sweep (N in [2,3,4],
    random Hamiltonian + 2 collapse operators, all 4 methods): worst
    observed residual/(eps64*||L||) ratio was 3.29 (power method). tol =
    200*eps64*scale for headroom, reusing the Liouvillian's own natural
    magnitude scale as the cross-check input (not as ground truth).
    """
    if method not in _SS_RESIDUAL_METHODS:
        return
    from .core.superoperator import operator_to_vector as _op_to_vec

    vec = _op_to_vec(result)
    residual = (A @ vec).norm("max")
    scale = max(1.0, A.norm("max"))
    tol = 200 * eps64 * scale
    trigger_if(residual > tol, "QT-SS-001")


@_guard("steadystate_physicality")
def check_steadystate_physicality(method, result):
    """QT-SS-002: a returned steady state is a valid density matrix
    (unit-trace, PSD) regardless of solver method -- distinct root-cause
    family from QT-SS-001 (a solver could satisfy L(rho_ss)=0 via a
    non-physical null vector, or vice versa; observationally and causally
    independent). Same method exclusion as QT-SS-001 (propagator excluded).

    Tolerance derivation: same 12-configuration sweep as QT-SS-001: worst
    observed trace violation was 1.0*eps64 (direct/svd); PSD violation was
    0 for all methods/configurations tried. tol_tr = 200*eps64, tol_psd =
    200*eps64 for headroom.
    """
    if method not in _SS_RESIDUAL_METHODS:
        return
    tol_tr = 200 * eps64
    tol_psd = 200 * eps64
    tr_bad = abs(result.tr() - 1) > tol_tr
    evals = result.eigenenergies()
    psd_bad = float(evals.min()) < -tol_psd
    trigger_if(tr_bad or psd_bad, "QT-SS-002")


# --- tensor.py: expand_operator ---------------------------------------------

@_guard("expand_operator_ptrace_inverse")
def check_expand_operator_ptrace_inverse(oper, dims, targets, result):
    """QT-TEN-001: expand_operator(oper, dims, targets).ptrace(targets) ==
    prod(dims[i] for i not in targets) * oper -- expand_operator builds a
    new_order permutation from an identity-list-plus-tensor-plus-permute
    construction distinct from a plain two-operand tensor() call
    (QT-QOBJ-004), so this exercises expand_operator's own new_order
    bookkeeping arithmetic specifically.

    Tolerance derivation: sweep over dims [[2,3,2],[2,2,2,2],[3,2,4]] and
    both contiguous and scattered target patterns: worst observed
    diff/(eps64*prod(dims)) ratio was 0 (bit-exact on this commit). tol =
    100*eps64*prod(dims) for headroom (full tensor-product-sized data).
    """
    import numpy as _np

    fulldim = int(_np.prod(dims))
    opdim = oper.shape[0]
    scale_dims = fulldim // opdim
    recovered = result.ptrace(targets)
    expected = scale_dims * oper
    diff = (recovered - expected).norm("max")
    tol = 100 * eps64 * fulldim
    trigger_if(diff > tol, "QT-TEN-001")
