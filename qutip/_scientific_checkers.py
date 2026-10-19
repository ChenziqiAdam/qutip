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

import contextlib
import json
import os
import threading
import warnings

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


def _tidy_atol():
    """User-requested sparse cleanup threshold (``auto_tidyup_atol``), or 0
    when ``auto_tidyup`` is off. Entries with abs <= this value are zeroed by
    the library on request, so a checker comparing against exact algebra must
    allow the error such a per-entry perturbation induces."""
    try:
        from qutip import settings as _s
        return float(_s.core["auto_tidyup_atol"]) if _s.core["auto_tidyup"] else 0.0
    except Exception:
        return 0.0


def _within(a, b, atol):
    """True when ``a`` and ``b`` differ by at most the absolute tolerance
    ``atol``. ``atol`` should be derived from a numerical sweep (SANITIZER.md
    5.8, T) and may legitimately exceed 1 for a magnitude-scaled quantity.
    """
    return abs(complex(a) - complex(b)) <= atol


@contextlib.contextmanager
def _quiet():
    """Checker re-calls must not leak warnings or floating-point errors to user code.

    The ignore filter is inserted into ``warnings.filters`` directly instead of
    using ``catch_warnings``: the latter invalidates every module's
    once-per-location warning registry, which would make production warnings
    repeat after each checker call.
    """
    flt = ("ignore", None, Warning, None, 0)
    warnings.filters.insert(0, flt)
    try:
        try:
            import numpy as _np
        except Exception:
            yield
        else:
            with _np.errstate(all="ignore"):
                yield
    finally:
        try:
            warnings.filters.remove(flt)
        except ValueError:
            pass


def _noncanonical(*objs):
    """True if any Qobj (or list/tuple of Qobjs) stores CSR data with
    duplicate or unsorted column indices. qutip kernels assume canonical
    CSR and disagree with each other on duplicate entries, so no
    cross-kernel invariant is meaningful for such an input."""
    for o in objs:
        if isinstance(o, (list, tuple)):
            if _noncanonical(*o):
                return True
            continue
        d = getattr(o, "data", None)
        if type(d).__name__ == "CSR":
            try:
                if not d.as_scipy().has_canonical_format:
                    return True
            except Exception:
                return True
    return False


def _narrow_scalars(args, kwargs):
    """Unified narrow-dtype policy (R12) for the scalar arguments of a checker.

    Qobj data is always complex128, so operands reach the checkers widened
    exactly; only user-supplied numpy *scalars* (N, offset, j, alpha, ...)
    keep a narrow dtype. Returns ``(args, kwargs, eps_in, wraps)``:
    numpy scalars become exact Python numbers (the checker's own tolerance /
    index arithmetic must not underflow in float16 or wrap in int8);
    ``eps_in`` is the eps of the narrowest floating scalar (>= eps64), for
    checkers whose library path computes in that dtype; ``wraps`` is True when
    a numpy integer scalar could wrap in the library's own sums/products
    (bound N*(N+1), N = sum |n|, against np.iinfo of the narrowest integer
    scalar), a
    dtype fact rather than a law violation.
    """
    import numpy as np

    eps_in, ints, imax = eps64, 0, None
    vals = list(args) + list(kwargs.values())
    for v in vals:
        if isinstance(v, np.inexact):
            eps_in = max(eps_in, float(np.finfo(v.dtype).eps))
        elif isinstance(v, np.integer):
            m = int(np.iinfo(v.dtype).max)
            imax = m if imax is None else min(imax, m)
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            ints += abs(int(v))
    # R13: the library forms n*(n+1) (jmat: j*(j+1); n^2 alone missed int8
    # j=11, 11*12 = 132 > 127), so bound that product, not (sum |n|)^2.
    wraps = imax is not None and ints * (ints + 1) > imax

    def py(v):
        return v.item() if isinstance(v, np.generic) and v.dtype.kind in "iufc" else v

    return (tuple(py(a) for a in args), {k: py(v) for k, v in kwargs.items()},
            eps_in, wraps)


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
                if _noncanonical(*args, *kwargs.values()):
                    return
                args, kwargs, _active.eps_in, wraps = _narrow_scalars(args, kwargs)
                if wraps:
                    return  # P: integer scalar may wrap in the library's arithmetic
                with _quiet():
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


def _dm_tol(rho):
    """Round-off tolerance of the valid-density-matrix gate (R12c), shared by
    every checker whose manifest precondition says "valid density matrix".

    Qobj data layers (Dense/CSR/Dia) all store complex128, so narrow user
    arrays are widened exactly at construction and eps64 is the right unit; a
    state that was *rounded* in float32 (trace off by ~d*eps32) is rejected
    here, which is intended -- every gated law has an eps64 tolerance. d
    entries are summed for the trace (each |rho_ij| <= 1 for a valid state),
    and the library's requested sparse cleanup perturbs each entry by <=
    tidy: tol = 20*eps64*d + d*tidy (derivation in check_purity_bound)."""
    d = rho.shape[0]
    return 20 * eps64 * d + d * _tidy_atol()


def _negative_mass(rho):
    """Total negative eigenvalue mass of the Hermitian part of ``rho``
    (eigensolver round-off on a PSD input is ~ d*eps*||rho||)."""
    import numpy as np
    a = rho.full()
    lam = np.linalg.eigvalsh((a + a.conj().T) / 2)
    return float(-lam[lam < 0].sum())


def _is_density_matrix(rho, psd=True):
    """True when ``rho`` is a valid density matrix within round-off: a square
    oper Qobj with |Tr rho - 1|, max|rho - rho^dag| and (``psd``) the negative
    eigenvalue mass all <= _dm_tol(rho). Kets/bras return False; a checker
    whose law is also stated for kets admits them itself. Scaled (2*rho,
    3*Bell), indefinite or non-Hermitian operators are outside the stated
    precondition of the gated laws, which then legitimately fail."""
    if getattr(rho, "type", None) != "oper" or rho.shape[0] != rho.shape[1]:
        return False
    tol = _dm_tol(rho)
    if abs(complex(rho.tr()) - 1) > tol:
        return False  # not unit trace
    if (rho - rho.dag()).norm("max") > tol:
        return False  # not Hermitian
    return not psd or _negative_mass(rho) <= tol


@_guard("purity_simplex_bound")
def check_purity_bound(dim, purity_value, rho=None):
    """QT-QOBJ-002: for a valid density matrix of dimension d,
    1/d <= purity <= 1.

    Precondition: operator-type Qobj only (ket/bra branch is a local
    tautology, out of scope per SANITIZER.md 5.6), and ``rho`` must be a
    density matrix. purity() returns Tr(A^2) for ANY operator (it neither
    normalizes nor validates), so a scaled, non-Hermitian or indefinite
    operator legitimately leaves [1/d, 1]; the hook passes ``rho`` and the
    gate below enforces the stated precondition (R12b; ``rho=None`` keeps the
    ungated form for direct calls).

    Gate tolerances (d entries summed, each |rho_ij| <= 1 for a valid state;
    the library's sparse cleanup perturbs each entry by <= tidy):
      - unit trace: |Tr rho - 1| <= tol_tr = 20*eps64*d + d*tidy, chosen so
        the induced shift of the bounds (<= 2*tol_tr/d resp. 2*tol_tr) stays
        below the purity tolerance;
      - Hermitian: max|rho - rho^dag| <= tol_tr (the anti-Hermitian part K
        lowers Tr(A^2) by ||K||_F^2, negligible at that size);
      - PSD (needed only for the upper bound, so the eigensolve runs only on
        the alarm path): negative eigenvalue mass n <= tol_tr (eigensolver
        round-off ~ d*eps*||rho||); Tr(A^2) <= ||A||_1^2 = (Tr A + 2n)^2, so
        the upper tolerance is widened by 4n.
    The lower bound needs only Hermitian + unit trace (Cauchy-Schwarz).
    """
    if dim < 1:
        return
    tidy = _tidy_atol()
    tol = 50 * eps64 * dim + 3 * dim * tidy
    p = purity_value.real if hasattr(purity_value, "real") else purity_value
    if rho is None:
        trigger_if(p < (1.0 / dim) - tol or p > 1.0 + tol, "QT-QOBJ-002")
        return
    if not _is_density_matrix(rho, psd=False):
        return  # P: not a unit-trace Hermitian operator
    if p < (1.0 / dim) - tol:
        trigger("QT-QOBJ-002")
    elif p > 1.0 + tol:
        n = _negative_mass(rho)
        if n > _dm_tol(rho):
            return  # P: not positive semidefinite
        trigger_if(p > 1.0 + tol + 4 * n, "QT-QOBJ-002")


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
    if norm_kind == "tr" and len(result.shape) == 2 and max(result.shape) <= 128:
        # Trace norm of a sparse operator is sum(sqrt(eig(M M^dag))); each
        # retained singular value s_i carries rounding error ~ eps*smax^2/(2 s_i)
        # (conditioning, not a defect). Numerically zero singular values are
        # excluded here, so a rank-deficient Dense operator (exact trace norm
        # via SVD) keeps the tight bound; sparse layers get the term below.
        import numpy as np
        sv = np.linalg.svd(result.full(), compute_uv=False)
        smax = float(sv.max()) if sv.size else 0.0
        if smax > 0:
            n = max(result.shape)
            pos = sv[sv > n * eps64 * smax]
            tol += 2 * eps64 * smax ** 2 * float(np.sum(1.0 / pos))
            if type(result.data).__name__ in ("CSR", "Dia"):
                # Sparse layers (trace_csr) sum sqrt(|eig(M M^dag)|): each of
                # the k numerically zero singular values becomes the sqrt of
                # an eigenvalue round-off of up to n*eps*smax^2, i.e.
                # sqrt(n*eps)*smax, in each of the two norm evaluations.
                # Maintainers treat this ~1e-8 error as expected (qutip#3025,
                # closed). Dense uses an exact SVD and keeps the tight bound.
                k = sv.size - pos.size
                tol += 2 * k * np.sqrt(n * eps64) * smax
                # R13: matmul_csr auto-tidies the Gram matrix M M^dag, zeroing
                # entries <= t = auto_tidyup_atol (absolute). That perturbs it
                # by ||E||_2 <= n*t, which moves sqrt(s_i^2) by at most
                # min(sqrt(n t), n t / (2 s_i)): a singular value 1e-7 (Gram
                # entry 1e-14) is dropped whole. Same family as qutip#3025.
                t = n * _tidy_atol()
                if t > 0:
                    tol += 2 * float(np.sum(np.minimum(
                        np.sqrt(t), t / (2 * np.maximum(sv, np.finfo(float).tiny)))))
    elif (norm_kind == "tr" and len(result.shape) == 2
          and type(result.data).__name__ in ("CSR", "Dia")):
        # Large sparse operator, no SVD: an eigenvalue error of n*eps*smax^2
        # moves any sqrt(lam_i) by at most sqrt(n*eps)*smax, so all n terms of
        # both evaluations are covered by 2*n*sqrt(n*eps)*smax, smax <= ||.||_F.
        import math
        n = max(result.shape)
        tol += 2 * n * math.sqrt(n * eps64) * float(result.norm("fro"))
        # R13: Gram-matrix auto-tidyup (see above), sqrt(n t) per eigenvalue.
        tol += 2 * n * math.sqrt(n * _tidy_atol())
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
    # Conditioning of the trace sum: the float summation error of Tr(B) is
    # ~eps*sum_k|b_kk|, so when the spectrum cancels (sum|b_kk| >> |Tr B|)
    # the tolerance must scale with sum|b_kk|*max|A|, not with |Tr B|*max|A|.
    import numpy as _np
    from .core import data as _qdata
    # Reference traces come from the data layer: Qobj.tr() keeps only the
    # real part whenever isherm passes at settings.core['atol'], which would
    # drop a sub-atol imaginary trace that ptrace correctly carries.
    trA = complex(_qdata.trace(A.data))
    trB = complex(_qdata.trace(B.data))
    absdiag_A = float(_np.abs(_np.diag(A.full())).sum())
    absdiag_B = float(_np.abs(_np.diag(B.full())).sum())
    rec0 = product.ptrace([0])
    exp0 = A * trB
    tol0 = 50 * eps64 * n * m * max(
        1.0, exp0.norm("max"), A.norm("max") * absdiag_B)
    tidy = 2 * n * m * _tidy_atol()
    tol0 += tidy
    trigger_if((rec0 - exp0).norm("max") > tol0, "QT-QOBJ-004")

    rec1 = product.ptrace([1])
    exp1 = B * trA
    tol1 = 50 * eps64 * n * m * max(
        1.0, exp1.norm("max"), B.norm("max") * absdiag_A)
    tol1 += tidy
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
    # expm of a generator of norm ~|alpha|*sqrt(N) loses ~|alpha|*eps (observed
    # 0.4*|alpha|*eps against a 60-digit reference); the 200*eps sweep was
    # taken at |alpha| <= 9, so scale linearly beyond |alpha| = 1.
    tol = 200 * eps64 * max(1.0, abs(alpha))
    tol += (2 * result.shape[0]) ** 0.5 * _tidy_atol()
    trigger_if(diff > tol, "QT-STA-002")


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
    action on deterministic generic density matrices in the Kraus operator's
    actual input space. The maximally mixed state alone is NOT sufficient probe
    coverage (SANITIZER.md 5.8, N): it is a fixed point of every unitary
    conjugation, so any two unitarily-related (or otherwise unital)
    channels agree on it even when they are genuinely different maps --
    confirmed during Step 4 sensitivity testing, 2026-09-16, where a
    deliberately mismatched Kraus/Choi pair (sigmax/sigmay vs. sigmaz
    channels) produced identical output on maximally_mixed_dm alone. Several
    seeded generic states are used so the probe can distinguish channels that
    merely happen to agree on one special input.

    FIX (2026-09-21, fresh-agent evaluation): the re-call intentionally uses
    to_kraus(tol=1e-12), so the comparison now budgets the documented
    discarded Choi weight rather than demanding epsilon-level equality after
    truncation. Generic seeded states support energy-restricted dimensions;
    checker-side exceptions return silently rather than becoming alarms.
    """
    from . import to_kraus as _to_kraus
    from .random_objects import rand_dm as _rand_dm

    if not kraus_ops:
        return
    dim_in = kraus_ops[0].shape[1]
    dim_out = kraus_ops[0].shape[0]
    if dim_in != dim_out:
        return
    try:
        recall_tol = 1e-12
        roundtrip_ops = _to_kraus(choi_result, tol=recall_tol)
        input_space = kraus_ops[0]._dims[1]
        # Build deterministic generic density matrices in the Kraus
        # operator's actual input space.  Flat basis(dim, i) states are not
        # dimension-compatible with energy-restricted spaces even when their
        # matrix shapes agree, and a maximally mixed state alone cannot
        # distinguish unital channels.
        test_states = [
            _rand_dm(input_space, seed=1729 + i)
            for i in range(min(dim_in + 1, 4))
        ]
        worst_diff = 0.0
        worst_scale = 1.0
        for rho in test_states:
            out1 = sum((k * rho * k.dag() for k in kraus_ops))
            out2 = sum((k * rho * k.dag() for k in roundtrip_ops))
            d = (out1 - out2).norm("max")
            if d > worst_diff:
                worst_diff = d
                worst_scale = max(1.0, out1.norm("max"))
        rounding_tol = 200 * eps64 * (dim_in ** 2) * worst_scale
        # to_kraus deliberately discards each Choi eigencomponent below
        # recall_tol.  Budget that documented approximation instead of
        # demanding machine-epsilon agreement after intentional truncation.
        truncation_tol = 2 * (dim_in ** 2) * recall_tol * worst_scale
        tol = max(rounding_tol, truncation_tol)
    except Exception:
        # A failed checker-side re-call is an instrumentation defect, not
        # evidence that the production channel conversion violated its law.
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

def _is_state_like(*qobjs):
    """True when every argument is a ket, bra or oper Qobj. The metric laws
    are stated for those types only; qutip's process_fidelity calls fidelity
    on super-typed Choi matrices, which are outside the stated precondition.
    """
    return all(getattr(q, "type", None) in ("ket", "bra", "oper") for q in qobjs)


def _met_cond_tol(A, B):
    """Extra absolute tolerance on F(A, B) explained by the conditioning of
    the eigenvalue-sqrt formulation: the error in sum(sqrt(lam_i)) of
    X = sqrt(A) B sqrt(A) is ~ eps*n/sqrt(lam_min). Applied only when lam_min
    is resolved above the eigenvalue round-off floor (lam_min >= 100*n*eps);
    numerically rank-deficient pairs (pure states, noise-level lam_min) return
    0, so the sqrt(eps) rank-deficient defect keeps firing."""
    import numpy as _np
    import math
    if getattr(A, "type", None) != "oper" or getattr(B, "type", None) != "oper":
        return 0.0
    a = A.full()
    b = B.full()
    n = a.shape[0]
    w, v = _np.linalg.eigh((a + a.conj().T) / 2)
    sa = (v * _np.sqrt(_np.clip(w, 0, None))) @ v.conj().T
    x = sa @ b @ sa
    lam = _np.linalg.eigvalsh((x + x.conj().T) / 2).min()
    if lam < 100 * n * eps64:
        return 0.0
    return 4 * eps64 * n / math.sqrt(lam)


def _met_dist_tol(A, B):
    """Trace-distance analogue: the library forms eig(D^dag D), whose zero
    eigenvalues cost sqrt(eps); only relevant when D = A - B has no (near-)
    zero eigenvalue, where the error is ~ eps*n/|d_min|."""
    import numpy as _np
    import math
    if getattr(A, "type", None) != "oper" or getattr(B, "type", None) != "oper":
        return 0.0
    d = _np.abs(_np.linalg.eigvalsh(A.full() - B.full())).min()
    n = A.shape[0]
    return 4 * eps64 * n / max(d, math.sqrt(eps64)) if d > 1e-8 else 0.0


@_guard("fidelity_symmetry")
def check_fidelity_symmetry(A, B, f_ab):
    """QT-MET-001: fidelity(A, B) == fidelity(B, A) -- symmetric by
    definition, but computed via argument-order-dependent code paths
    (an internal swap "for numerical stability").
    """
    from . import fidelity as _fidelity

    if not _is_state_like(A, B):
        return
    # R12c: manifest -- kets/bras any, operator arguments valid density
    # matrices (an indefinite Hermitian A makes sqrtm(A) complex and the
    # swapped evaluation legitimately differs).
    if not all(q.type != "oper" or _is_density_matrix(q) for q in (A, B)):
        return
    f_ba = _fidelity(B, A)
    tol = 500 * eps64 + 2 * _met_cond_tol(A, B)
    trigger_if(abs(f_ab - f_ba) > tol, "QT-MET-001")


@_guard("fidelity_self_identity")
def check_fidelity_self_identity(A):
    """QT-MET-002: fidelity(A, A) == 1 for any valid density matrix,
    specifically exercising the operator (sqrtm+eigenvalue) branch.
    """
    from . import fidelity as _fidelity

    if not _is_density_matrix(A):
        return  # R12c: F(2 rho, 2 rho) = 2 legitimately
    f_aa = _fidelity(A, A)
    tol = 2000 * eps64 + 2 * _met_cond_tol(A, A)
    trigger_if(abs(f_aa - 1) > tol, "QT-MET-002")


@_guard("tracedist_metric_axioms")
def check_tracedist_metric_axioms(A, B, d_ab, sparse=False, tol=0):
    """QT-MET-003: tracedist is a proper metric -- zero iff equal,
    symmetric, obeys the triangle inequality. Re-calls tracedist on (A, A),
    (B, A), and against a fresh random-ish comparator to probe the triangle
    inequality using the already-available B as the midpoint.

    Re-calls use the caller's ``sparse``/``tol`` (a dense re-call would
    compare LAPACK against ARPACK). The dense path is deterministic and
    D^dag D is bit-identical for A-B and B-A, so symmetry stays exact. The
    sparse path is a randomly-started iterative solve: each of the n
    eigenvalues of D^dag D carries an absolute error up to
    delta = (n*eps + max(tol, eps)) * lam_max, and the library sums
    sqrt(|lam|), so a numerically-zero eigenvalue may contribute sqrt(delta)
    to each of two independent runs: |d_ab - d_ba| <= n*sqrt(delta).
    lam_max is bounded by ||A - B||_F^2. D(A, A) has D = 0 exactly on both
    paths, so the self-distance alarm keeps the fixed tolerance.
    """
    from . import tracedist as _tracedist
    import math

    if not _is_state_like(A, B):
        return
    atol = 100 * eps64
    trigger_if(abs(_tracedist(A, A, sparse=sparse, tol=tol)) > atol, "QT-MET-003")
    d_ba = _tracedist(B, A, sparse=sparse, tol=tol)
    sym_tol = atol
    if sparse:
        P, Q = (q.proj() if q.type in ("ket", "bra") else q for q in (A, B))
        n = P.shape[0]
        lam_max = float((P - Q).norm("fro")) ** 2
        delta = (n * eps64 + max(float(tol), eps64)) * lam_max
        sym_tol += n * math.sqrt(delta)
    trigger_if(abs(d_ab - d_ba) > sym_tol, "QT-MET-003")


@_guard("fuchs_van_de_graaf_bound")
def check_fuchs_van_de_graaf(A, B, f_ab):
    """QT-MET-004: 1 - F(A,B) <= D(A,B) <= sqrt(1 - F(A,B)^2), the
    Fuchs-van de Graaf inequalities relating fidelity and trace distance --
    a cross-quantity theorem, not a restatement of either function's own
    code.
    """
    from . import tracedist as _tracedist
    import math

    if not (_is_density_matrix(A) and _is_density_matrix(B)):
        return  # R12c: the bounds hold for density matrices only (hook passes projected kets)
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

    if not (_is_density_matrix(A) and _is_density_matrix(B)):
        return  # R12c: manifest precondition (hook passes projected kets)
    dim = A.shape[0]
    try:
        U = rand_unitary(dim)
        A2 = U * A * U.dag()
        B2 = U * B * U.dag()
    except Exception:
        return
    d1, d2 = _bures_dist(A, B), _bures_dist(A2, B2)
    f1 = _fidelity(A, B)
    # fidelity error tF (conditioning) is amplified by the sqrt in the
    # distance/angle near F = 1: d ~ sqrt(2(1-F)), angle ~ arccos(F).
    import math
    tF = 2 * (_met_cond_tol(A, B) + _met_cond_tol(A2, B2))
    tol_d = 2000 * eps64 + tF / max(d1, math.sqrt(max(tF, eps64)))
    trigger_if(abs(d1 - d2) > tol_d, "QT-MET-005")
    if abs(f1 - 1) > 1e-6:
        a1, a2 = _bures_angle(A, B), _bures_angle(A2, B2)
        sn = math.sqrt(max(1 - min(1.0, f1) ** 2, 0.0))
        tol_a = 2000 * eps64 + tF / max(sn, math.sqrt(max(tF, eps64)))
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
    # R12: jmat builds its matrix elements in j's own dtype (float16 j=100:
    # J+ relative error 1.4e-3 = 2.9 eps16), so use the narrow scalar eps
    eps = getattr(_active, "eps_in", eps64)
    tol = 50 * eps * scale * max(1.0, j)
    for (A, B, C) in ((Jx, Jy, Jz), (Jy, Jz, Jx), (Jz, Jx, Jy)):
        comm = A @ B - B @ A
        diff = (comm - 1j * C).norm("max")
        trigger_if(diff > tol, "QT-OPS-001")


@_guard("ladder_operator_ccr")
def check_ladder_ccr(N, result_destroy, result_create, offset=0):
    """QT-OPS-002: [a, a^dagger] = I on the interior (Fock levels 0..N-2)
    of a finite truncation; the top level (N-1) is the well-known finite-
    truncation boundary artifact and is explicitly excluded from the
    precondition, not silently special-cased.

    With a nonzero ``offset`` the ladder acts on levels offset..offset+N-1
    and [a, a^dagger]_00 = offset+1 (the lowest level has no lower
    neighbour); that bottom boundary level is excluded as well.

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
    lo = 0 if offset == 0 else 1
    if N - 1 <= lo:
        return
    block = full[lo: N - 1, lo: N - 1]
    diff = float(_np.abs(block - _np.eye(N - 1 - lo)).max())
    # Entries of a a^dagger are ~(offset + k): float rounding of sqrt(k+offset)
    # products is ~offset*eps (measured ~0.5*offset*eps up to offset=1e9), so
    # the O(N) accumulation term is joined by an O(offset) one.
    tol = 50 * eps64 * (N + offset)
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
def check_expm_generator_unitary(N, magnitude, result, checker_id,
                                 offset=0, kind="displace"):
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
    # R12c: manifest precondition -- the tolerance was derived on the swept
    # range |alpha|, |z| in [0.1, 15] only; outside it the law is not checked.
    if not 0.1 <= abs(magnitude) <= 15:
        return
    # Dense numpy arithmetic so the library's requested cleanup does not also
    # tidy the checker's own product.
    import numpy as np
    M = result.full()
    diff = np.abs(M.conj().T @ M - np.eye(N)).max()
    # Unitarity loss of expm is ~||G||_2*eps. ||G|| = O(|alpha|*sqrt(offset+N))
    # for displace and O(|z|*(offset+N)) for squeeze (a^2 has norm ~offset+N);
    # both reduce to |magnitude| (relative to the N-sweep) at offset=0.
    if kind == "squeeze":
        gscale = abs(magnitude) * (offset + N) / N
    else:
        gscale = abs(magnitude) * ((offset + N) / N) ** 0.5
    tol = 50 * eps64 * N * max(1.0, abs(magnitude), gscale)
    # A requested per-entry cleanup (auto_tidyup_atol = t) perturbs U by E with
    # |E_ij| <= t, which changes U^dag U - I by at most 2*sqrt(N)*t + N*t^2.
    t = _tidy_atol()
    tol += 2 * N ** 0.5 * t + N * t * t
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
    from .core.dimensions import Compound, Space, SuperSpace

    dims0 = q_oper._dims[0]
    if not isinstance(dims0, Compound):
        return
    # The involution is well-posed only when every factor is a superoperator
    # on one elementary space of dimension >= 2. A size-1 ("trivial ancilla")
    # space is dropped by the dims bookkeeping, so the reshuffled object
    # cannot be re-split (TypeError) although its matrix is correct; a
    # compound factor ([2,2] x [3]) is regrouped into elementary factors by
    # the first reshuffle, so the second one targets a different partition.
    # Neither is a violation of the relabeling law.
    for space in dims0.spaces:
        oper = space.oper if isinstance(space, SuperSpace) else None
        if oper is None or type(oper.to_) is not Space or oper.to_.size < 2:
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
    import numpy as _np

    # Dense numpy product: the checker's own sparse product must not be
    # tidied by the library's requested cleanup.
    M = result.full()
    diff = float(_np.abs(M @ M.conj().T - _np.eye(N)).max())
    tol = 50 * eps64 * N
    # A requested per-entry cleanup (auto_tidyup_atol = t) perturbs U by E with
    # |E_ij| <= sqrt(2) t, which changes U U^dag - I by at most
    # 2 sqrt(N) |E|max + N |E|max^2.
    t = 2 ** 0.5 * _tidy_atol()
    tol += 2 * N ** 0.5 * t + N * t * t
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
    tidy = 2 ** 0.5 * N * _tidy_atol()
    tol_tr += tidy
    tol_psd += tidy
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
    tol += len(result) * (2 * 2 ** 0.5 * N + 1) * _tidy_atol()
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
    # A requested cleanup drops at most N entries of magnitude <= t per
    # row/column.
    tol += N * _tidy_atol()
    trigger_if(diff > tol, "QT-RND-004")


@_guard("bcsz_trace_preservation")
def check_bcsz_trace_preservation(N, D_pre_permute, Y_pre=None):
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
    if Y_pre is not None:
        # Error of Y^(-1/2) renormalisation scales with cond(Y) (rank-1 draws
        # reach cond ~6e4 while the base sweep saw cond <~ 100): scale beyond.
        tol *= max(1.0, float(_np.linalg.cond(_np.asarray(Y_pre))) / 100.0)
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
    import numpy as _np

    diff = float(_np.abs(dense.full() - sparse.full()).max())
    # The two containers are tidied independently when cleanup is requested:
    # per-entry difference <= sqrt(2) t (real and imaginary parts separately).
    tol += 2 ** 0.5 * _tidy_atol()
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
    # A requested cleanup drops at most dim diagonal entries of size <= sqrt(2) t.
    tol += 2 ** 0.5 * dim * _tidy_atol()
    from .core import data as _qdata
    # raw data-layer traces (Qobj.tr() drops a sub-atol imaginary part
    # for operators that pass isherm, and the two operands may differ there)
    diff = abs(complex(_qdata.trace(result.data))
               - complex(_qdata.trace(rho.data)))
    trigger_if(diff > tol, "QT-PT-002")


# --- entropy.py: quantum information measures -------------------------------

@_guard("concurrence_bound")
def check_concurrence_bound(result, rho=None):
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
    # R12c: the hook passes the (ket2dm-promoted) state; the bound holds for
    # valid density matrices only (concurrence(3*Bell) = 9). rho=None keeps
    # the ungated form for direct calls.
    if rho is not None and not _is_density_matrix(rho):
        return
    tol = 50 * eps64
    val = result.real if hasattr(result, "real") else result
    trigger_if(val > 1.0 + tol, "QT-ENT-001")


def _fannes(dims, sel, t, base):
    """Largest entropy change (in units of ``base``) of the marginal on the
    subsystems ``sel`` of a state with dimensions ``dims`` when every entry
    is perturbed by at most sqrt(2) t (Fannes-Audenaert bound)."""
    import math
    try:
        sel = list(sel)
    except TypeError:
        sel = [sel]
    d = 1
    for i in sel:
        d *= int(dims[i])
    if d <= 1:
        return 0.0
    T = 0.5 * d * d * 2 ** 0.5 * t
    if T >= 1.0 - 1.0 / d:
        f = math.log(d)
    else:
        h = -T * math.log(T) - (1 - T) * math.log(1 - T) if T > 0 else 0.0
        f = T * math.log(d - 1) + h
    return f / math.log(base)


@_guard("mutual_information_bounds")
def check_mutual_information_bounds(rho, selA, selB, result, base=None):
    """QT-ENT-002: entropy_mutual(rho,A,B) >= 0 (subadditivity) and
    <= 2*min(S(A),S(B)) -- neither bound is locally guaranteed by the
    three-term subtraction; each of S(A), S(B), S(AB) is computed via an
    independent eigenenergies() call with its own numerical error, and
    subadditivity is a deep theorem, not an algebraic identity.

    FIX (2026-09-21, round-1 triggerability): `base` must be threaded
    through from the entropy_mutual() call site to this checker and reused
    for the cross-check S(A)/S(B) computation. `result` (the mutual
    information) is computed by entropy_mutual() in whatever `base` the
    caller passed (default e, but e.g. 2 for bits); the bound
    `2*min(S(A),S(B))` scales linearly with the same log base, since
    S(A)=-sum(p*log_base(p)) rescales uniformly. The original code called
    `entropy_vn(ptrace(rho, selA))` with entropy_vn's own default
    (`base=e`) regardless of what base `result` was computed in, so a
    base=2 mutual information (in bits) was compared against a base-e
    (nats) bound -- units mismatch, not a real subadditivity violation.
    Confirmed directly: entropy_mutual(bell_state('00').proj(), [0], [1],
    base=2) gives result=2.0 exactly (the Bell state saturates the bound);
    the base-e marginal entropies are ln(2)=0.693 each, giving a wrong
    bound of 2*0.693=1.386 < 2.0 (spurious fire); the base-2 marginal
    entropies are log2(2)=1.0 each, giving the correct bound
    2*1.0=2.0 >= 2.0 (satisfied, within tolerance). `base=None` (e.g. a
    stale caller that hasn't been updated) falls back to `np.e` to match
    entropy_mutual's own default and preserve prior behavior for those
    call sites.

    Tolerance derivation: sweep over dims [[2,2],[2,3],[3,3],[2,2,2]], 30
    trials each, selA=[0]/selB=rest, bases {e, 2}: worst observed
    lower-bound slack (most negative MI) was 0; worst upper-bound excess
    was 0 (no violation observed at any conditioning or base tried once
    the base mismatch above was fixed). tol = 200*eps64*cond, cond taken
    as max(1, 1/min_nonzero_eigenvalue-scale) approximated here by a fixed
    generous constant since a tight cond estimate would require
    re-deriving each entropy call's own conditioning. A fresh-agent product
    of two rank-one six-level states exposed 8.13e-14 cancellation error
    against the old 4.44e-14 flat bound, so the bound now scales by total
    Hilbert-space dimension (36 for that witness).
    """
    import numpy as _np
    from .core.qobj import ptrace as _ptrace
    from .entropy import entropy_vn as _entropy_vn

    if not _is_density_matrix(rho):
        return  # R12c: subadditivity needs a valid density matrix (3*Bell: I < 0)
    if base is None:
        base = _np.e
    # Three entropy eigensolves are subtracted, and their absolute roundoff
    # grows with Hilbert-space dimension, especially for rank-deficient pure
    # states whose true entropies are zero.
    tol = 200 * eps64 * max(1, rho.shape[0])
    sA = _entropy_vn(_ptrace(rho, selA), base)
    sB = _entropy_vn(_ptrace(rho, selB), base)
    t = _tidy_atol()
    if t > 0:
        # A requested cleanup perturbs each marginal by E with |E_ij| <=
        # sqrt(2) t, i.e. trace distance T <= d^2 t / sqrt(2); entropy changes
        # by at most the Fannes-Audenaert bound (non-Lipschitz at zero
        # eigenvalues, so a pure marginal may become mixed).
        fA = _fannes(rho.dims[0], selA, t, base)
        fB = _fannes(rho.dims[0], selB, t, base)
        tol += fA + fB + 2 * max(fA, fB)
    lower_bad = result < -tol
    upper_bad = result > 2 * min(sA, sB) + tol
    trigger_if(lower_bad or upper_bad, "QT-ENT-002")


# --- solver/steadystate.py: steady-state solvers ----------------------------

# "iterative" is excluded like "propagator": its accuracy is set by the
# caller's tol/atol/rtol (default residual ~3e-5), so a fixed floating-point
# tolerance would only re-observe the user's own stopping criterion.
_SS_RESIDUAL_METHODS = ("direct", "eigen", "svd", "power")


def _ss_size_factor(n):
    """Round-off growth with the Hilbert dimension n. The original sweep
    covered n <= 4 only; a later sweep to n = 60 (oscillator and random
    Liouvillians, 4 methods) gave worst ratios to eps of 1771 (eigen PSD),
    1116 (direct trace) and 556 (power residual) at n = 60, and 4700 (eigen
    PSD) in the n = 30 upstream test, so the 200*eps base tolerance is
    multiplied by (n/4)**2 for n > 4."""
    return max(1.0, (n / 4.0) ** 2)


_SS_FWD_CAP = 1e-6


def _ss_conditioning(A, method, n):
    """Forward-error tolerance for a steady state of Liouvillian A. The
    returned rho solves L rho = 0 only up to the conditioning of L:
    forward error ~ eps*kappa (kappa = ||L|| / sigma_gap, sigma_gap the
    smallest nonzero singular value), and ~ eps*kappa^2 for method="eigen",
    which diagonalises L^dag L. Capped at an absolute accuracy of 1e-6 so
    total-loss results (error >> 1e-6) always fire. Returns (tol,
    degenerate); degenerate is True when L has numerical nullity > 1.
    With more than 900 rows the dense SVD is skipped (no conditioning
    relief, degenerate False)."""
    import numpy as _np
    base = 200 * eps64 * _ss_size_factor(n)
    if A.shape[0] > 900:
        return base * max(1.0, A.norm("max")), False
    sv = _np.linalg.svd(A.full(), compute_uv=False)
    gap = sv[-2]
    degenerate = bool(gap <= sv[0] * A.shape[0] * eps64)
    kappa = sv[0] / max(gap, 1e-300)
    p = 2 if method == "eigen" else 1
    tol = min(max(base * max(1.0, A.norm("max")), base * kappa ** p),
              _SS_FWD_CAP)
    return tol, degenerate


@_guard("steadystate_physicality")
def check_steadystate_physicality(A, method, result):
    """QT-SS-002: a returned steady state is a valid density matrix
    (unit-trace, PSD) regardless of solver method, for a Liouvillian A
    with a unique stationary state (numerical nullity 1). Methods
    "propagator" and "iterative" are excluded: their accuracy is set by
    the caller's own stopping criterion.

    Tolerance derivation: float64 sweeps (n in [2, 60], random and
    oscillator Liouvillians, 4 methods): worst trace violation 1116*eps64
    (direct, n = 60), worst PSD violation 1771*eps64 (eigen, n = 60), 4700
    (eigen, n = 30 upstream test). tol = 200*eps64*max(1,(n/4)**2),
    raised to the conditioning tolerance of L (see _ss_conditioning).
    """
    if method not in _SS_RESIDUAL_METHODS:
        return
    tol_fwd, degenerate = _ss_conditioning(A, method, result.shape[0])
    if degenerate:
        # non-unique stationary state: any null vector is a valid
        # "steady state" of L, so physicality is not defined for it
        return
    tol_tr = max(200 * eps64 * _ss_size_factor(result.shape[0]), tol_fwd)
    tol_psd = tol_tr
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

    # ptrace returns the kept subsystems in ascending order, so the identity
    # below only applies when the targets are given in ascending order.
    target_list = [targets] if isinstance(targets, (int, _np.integer)) else list(targets)
    if target_list != sorted(set(target_list)):
        return
    fulldim = int(_np.prod(dims))
    opdim = oper.shape[0]
    scale_dims = fulldim // opdim
    recovered = result.ptrace(targets)
    expected = scale_dims * oper
    diff = (recovered - expected).norm("max")
    # Absolute error of the recovered operator is relative to its magnitude.
    tol = 100 * eps64 * fulldim * max(1.0, expected.norm("max"))
    # A requested cleanup (entries <= t zeroed) drops oper entries before the
    # trace over scale_dims copies, and tidies both compared operators:
    # per-entry deviation <= (scale_dims + 2) sqrt(2) t.
    tol += 2 ** 0.5 * (scale_dims + 2) * _tidy_atol()
    trigger_if(diff > tol, "QT-TEN-001")
