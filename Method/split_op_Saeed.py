"""The user's original split_op_Saeed.py, adapted as closely as possible to:

  * TWO material regions on a 2D (Nx,Ny) lattice,
  * ONE coherent exciton-photon excitation, no molecule,
  * separate fitted multi-mode J1(w), J2(w) harmonic baths and MFE forces.

Original-style function names: Bunch, Epol, Enoncouplefunc, Ushuf,
Ushufrev, BlocksED, propagateCi, Force, VelVer, pop, runTraj.

ci is (2,Nx,Ny): ci[0] is exciton, ci[1] is photon, BOTH in REAL SPACE.
R,P,omega,c are (Nx,Ny,n_modes).  We use the full cavity momentum grid,
so there are no excluded photon k-modes and no dark exciton-only blocks.

Crucial correction relative to uploaded original:
    V^dagger * psi -> exp(-i*E*dt) -> V * psi
and NEVER V * psi -> exp(-i*E*dt) -> V^dagger * psi.

Electronic Strang: local(dt/2) / homogeneous polariton(dt) / local(dt/2).
MFE: half electronic / full nuclear velocity Verlet / half electronic.
"""

import numpy as np
from Model.polaritonmod import parameters, initR, EV


# ---------------------------------------------------------------------------
# ORIGINAL function retained: Bunch
# ---------------------------------------------------------------------------
class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


# ---------------------------------------------------------------------------
# ORIGINAL function retained: Epol
# ---------------------------------------------------------------------------
def Epol(param):
    """LP/UP eigenvalues and eigenvectors on the 2D reference k grid.

    Returns Epl shape (Nx,Ny), U shape (2,Nx,Ny).  Exactly the original
    Epol idea, but 1D momentum index k becomes (kx,ky).
    """
    E2, V2 = BlocksED(param)
    branch = getattr(param, 'branch', 0)
    if branch not in (0, 1, -1):
        raise ValueError('branch should be 0 (LP) or 1/-1 (UP)')
    Epl = E2[..., branch]
    U = np.moveaxis(V2[..., :, branch], -1, 0)
    return Epl, U


# ---------------------------------------------------------------------------
# ORIGINAL function retained: Enoncouplefunc
# ---------------------------------------------------------------------------
def Enoncouplefunc(parameters):
    """Empty because the 2D model uses FULL photon coupling for all k.

    Original 1D SQD truncated photon k modes and propagated leftover bare
    exciton k's separately.  Our 2D model has N_Pht=Nsite, so no such modes.
    """
    return np.empty((0,), dtype=float)


# ---------------------------------------------------------------------------
# ORIGINAL functions retained: Ushuf / Ushufrev
# ---------------------------------------------------------------------------
def Ushuf(parameters, ci):
    """From k-space fields (2,Nx,Ny) to 2x2-block order (Nsite,2).

    Call this AFTER fft2 on both real-space fields.  It is just a unitary
    axis permutation/reshape, replacing the original masked 1D shuffling.
    """
    expected = (2, parameters.Nx, parameters.Ny)
    if np.shape(ci) != expected:
        raise ValueError(f'Ushuf expected {expected}, got {np.shape(ci)}')
    return np.moveaxis(ci, 0, -1).reshape(parameters.Nsite, 2).copy()


def Ushufrev(parameters, ci):
    """Reverse block order (Nsite,2) -> k-space fields (2,Nx,Ny)."""
    expected = (parameters.Nsite, 2)
    if np.shape(ci) != expected:
        raise ValueError(f'Ushufrev expected {expected}, got {np.shape(ci)}')
    return np.moveaxis(np.asarray(ci).reshape(parameters.Nx, parameters.Ny, 2), -1, 0).copy()


# ---------------------------------------------------------------------------
# ORIGINAL function retained: BlocksED
# ---------------------------------------------------------------------------
def BlocksED(parameters):
    """Diagonalize ALL 2x2 homogeneous exciton-photon blocks once.

    Return shapes: (Nx,Ny,2) and (Nx,Ny,2,2), matching the previous 2D
    implementation. In create_dat we flatten these to original Saeed-style
    cached dat.E2_list (Nsite,2) and dat.V2_list (Nsite,2,2).
    Eigenvector columns are LP and UP.

    H0(q) = [[eps_k(q), g_k(q)], [g_k(q), omega_k(q)]].
    Spatial heterogeneity and local baths are NOT included in these blocks.
    """
    Nsite = parameters.Nsite
    εk = parameters.eps_k.reshape(Nsite)
    ωk = parameters.omega_k.reshape(Nsite)
    gk = parameters.g_k.reshape(Nsite)

    H2 = np.zeros((Nsite, 2, 2), dtype=np.complex128)
    H2[:, 0, 0] = εk
    H2[:, 1, 1] = ωk
    H2[:, 0, 1] = gk
    H2[:, 1, 0] = gk

    E2_list, V2_list = np.linalg.eigh(H2)
    return (E2_list.reshape(parameters.Nx, parameters.Ny, 2),
            V2_list.reshape(parameters.Nx, parameters.Ny, 2, 2))


# ---------------------------------------------------------------------------
# Compatibility with the first 2D version: optional Ugamma / Uk helpers.
# The uploaded 1D Saeed code performs both operations inside propagateCi().
# ---------------------------------------------------------------------------
def Ugamma(parameters, dat, ci, dt):
    """Isolated local exponential; dt is the requested COMPLETE substep."""
    ϵ0 = (dat.eps - parameters.reference_eps_ev * EV
          + np.einsum('xyj,xyj->xy', dat.c, dat.R))
    out = np.asarray(ci, dtype=np.complex128).copy()
    out[0] *= np.exp(-1j * ϵ0 * dt)
    return out


def Uk(parameters, ci, dt, dat=None):
    """Isolated reference polariton step: FFT2, V^dagger, phase, V, IFFT2.

    Calling with dat uses cached E2_list/V2_list. Calling without dat also
    works but diagonalizes the reference 2x2 blocks once for that call.
    """
    if dat is None:
        E2, V2 = BlocksED(parameters)
        E2_list = E2.reshape(parameters.Nsite, 2)
        V2_list = V2.reshape(parameters.Nsite, 2, 2)
    else:
        if not hasattr(dat, 'E2_list') or not hasattr(dat, 'V2_list'):
            E2, V2 = BlocksED(parameters)
            dat.E2_list = E2.reshape(parameters.Nsite, 2)
            dat.V2_list = V2.reshape(parameters.Nsite, 2, 2)
        E2_list, V2_list = dat.E2_list, dat.V2_list
    ct_trot = np.fft.fft2(ci, axes=(-2,-1), norm='ortho')
    ct_shuf = Ushuf(parameters, ct_trot)
    cres3 = np.einsum('nia,ni->na', V2_list.conj(), ct_shuf)
    cres4 = np.exp(-1j * E2_list * dt) * cres3
    cres5 = np.einsum('nia,na->ni', V2_list, cres4)
    return np.fft.ifft2(Ushufrev(parameters, cres5), axes=(-2,-1), norm='ortho')


# ---------------------------------------------------------------------------
# ORIGINAL function retained: propagateCi
# ---------------------------------------------------------------------------
def propagateCi(parameters, dat, ci, dt):
    """Original Saeed split-op layout, now fft2 over two spatial axes.

    REAL local half-step; FFT2 exciton AND photon; reorder to 2x2 blocks;
    V^dagger -> eigenphases -> V; reverse ordering; IFFT2 both; REAL local
    half-step.  The two-material phonon correction is site-dependent.
    """
    Nsite = parameters.Nsite
    ct_trot = np.asarray(ci, dtype=np.complex128).copy()
    if ct_trot.shape != (2, parameters.Nx, parameters.Ny):
        raise ValueError('ci must have shape (2,Nx,Ny)')

    # --- Original "gamma * R" real-space half-step, now for J1 and J2. ---
    # IMPORTANT: subtract the reference onsite energy already in eps_k(q).
    # In model eps_k = eps_ref -2*tau_ref*(cos kx + cos ky), so this is exact.
    ϵ0 = (dat.eps - parameters.reference_eps_ev * EV
          + np.einsum('xyj,xyj->xy', dat.c, dat.R))
    ct_trot[0] *= np.exp(-1j * (0.5 * ϵ0 * dt))

    # --- Original 1D exciton fft -> NEW 2D fft for exciton and photon. ---
    ct_trot = np.fft.fft2(ct_trot, axes=(-2, -1), norm='ortho')
    ct_shuf = Ushuf(parameters, ct_trot)

    # --- Original cres1..cres5 block evolution, now full 2D photon grid. ---
    # dat.E2_list and dat.V2_list are computed once by create_dat/runTraj.
    if not hasattr(dat, 'E2_list') or not hasattr(dat, 'V2_list'):
        E2, V2 = BlocksED(parameters)
        dat.E2_list = E2.reshape(Nsite, 2)
        dat.V2_list = V2.reshape(Nsite, 2, 2)
    cres1 = ct_shuf.reshape(Nsite, 2)
    cres2 = cres1.copy()

    # FIXED order: transform *into* eigenbasis with V^dagger = conj(V).T.
    # Note einsum sums the input-component index i:
    #   cres3[n, a] = sum_i conj(V[n,i,a]) cres2[n,i].
    cres3 = np.einsum('nia,ni->na', dat.V2_list.conj(), cres2)
    cres4 = np.exp(-1j * dat.E2_list * dt) * cres3

    # Transform *back* to exciton/photon representation using V.
    cres5 = np.einsum('nia,na->ni', dat.V2_list, cres4)

    # Here all photon k-modes are represented: no Enoncouple phase, masked
    # shuffle reconstruction, or dark exciton correction (Delta c) required.
    ct_shuf = cres5
    ct_trot = Ushufrev(parameters, ct_shuf)
    ct_trot = np.fft.ifft2(ct_trot, axes=(-2, -1), norm='ortho')

    # --- Original real-space local second half-step. ---
    ct_trot[0] *= np.exp(-1j * (0.5 * ϵ0 * dt))
    return ct_trot


# ---------------------------------------------------------------------------
# ORIGINAL function retained: Force
# ---------------------------------------------------------------------------
def Force(par, dH0=None, ci=None, c=None):
    """-dH0 - c_site,mode * |ci_ex(site)|^2 (one excitation).

    Original: Force(par,dH0,ci) with a single scalar gamma.
    New: Force(par,dH0,ci,c), where c has shape (Nx,Ny,n_modes).
    For compatibility with the first 2D project, Force(dat) also works.
    """
    if dH0 is None and ci is None and hasattr(par, 'param'):
        dat = par
        par, dH0, ci, c = dat.param, dat.omega**2*dat.R, dat.ci, dat.c
    if dH0 is None or ci is None or c is None:
        raise TypeError('Use Force(par,dH0,ci,site_c) or Force(dat)')
    F = -np.array(dH0, dtype=float, copy=True)
    F -= c * np.abs(ci[0])[..., None]**2
    return F


# ---------------------------------------------------------------------------
# ORIGINAL function retained: VelVer
# ---------------------------------------------------------------------------
def VelVer(dat):
    """Same original electronic / nuclear / electronic layout.

    The original odd EStep=5 splits into 2+3 steps, which are unequal.
    Here EStep=2*e_substeps_half is even, so the two electronic halves are
    exactly dtN/2.  The nuclear forces use the same frozen midpoint ci.
    """
    par = dat.param
    M = float(getattr(par, 'M', 1.0))  # mass-weighted Q/P -> M=1
    if M <= 0:
        raise ValueError('M must be positive')
    v = dat.P/M
    ci = dat.ci.copy()

    EStep = 2*par.e_substeps_half
    dtE = par.dtN/EStep

    # -------- Electronic half-step at OLD nuclear coordinates --------------
    for t in range(EStep//2):
        ci = propagateCi(par, dat, ci, dtE)
    dat.ci = ci.copy()

    # -------- Nuclear block: original velocity Verlet ---------------------
    # IMPORTANT: refresh F1 AFTER the electronic half-step; it must be
    # evaluated with the frozen midpoint electronic state, not pre-midpoint.
    dat.dH0 = dat.omega**2 * dat.R
    F1 = Force(par, dat.dH0, dat.ci, dat.c)
    dat.R += v*par.dtN + 0.5*F1*par.dtN**2/M
    dat.dH0 = dat.omega**2 * dat.R
    F2 = Force(par, dat.dH0, dat.ci, dat.c)
    v += 0.5*(F1 + F2)*par.dtN/M
    dat.P = v*M
    dat.F1 = F2  # original diagnostic/cache field, not reused next step

    # -------- Electronic half-step at NEW nuclear coordinates --------------
    for t in range(EStep//2):
        ci = propagateCi(par, dat, ci, dtE)
    dat.ci = ci.copy()
    return dat


# ---------------------------------------------------------------------------
# ORIGINAL function retained: pop
# ---------------------------------------------------------------------------
def pop(dat):
    """Full single-excitation density matrix, like original SQD pop().

    WARNING: O((2*Nsite)^2) memory. Use |ci[0]|^2 and |ci[1]|^2 for
    the space-resolved interference signal, not this huge matrix.
    """
    ci = dat.ci.reshape(-1)
    return np.outer(ci.conj(), ci)


def total_norm(ci):
    return float(np.sum(np.abs(ci)**2))


def energy(dat):
    """Conserved continuum Hamiltonian expectation for testing convergence."""
    par = dat.param
    Xk = np.fft.fft2(dat.ci[0], norm='ortho')
    Ck = np.fft.fft2(dat.ci[1], norm='ortho')
    e0 = np.sum(par.eps_k*np.abs(Xk)**2 + par.omega_k*np.abs(Ck)**2)
    e0 += 2*np.sum(par.g_k*np.real(Xk.conj()*Ck))
    ϵ0 = (dat.eps - par.reference_eps_ev*EV
          + np.einsum('xyj,xyj->xy', dat.c, dat.R))
    local = np.sum(ϵ0*np.abs(dat.ci[0])**2)
    phonon = 0.5*np.sum(dat.P**2 + (dat.omega*dat.R)**2)
    return float(e0 + local + phonon)


def create_dat(par, case, init, seed):
    """Build original Bunch with 2D/2-bath arrays; cache BlocksED once."""
    omega, c, eps = par.site_bath(case)
    R, P = initR(par, seed, case)
    dat = Bunch(param=par, case=case, ci=np.asarray(init).copy(),
                R=R, P=P, omega=omega, c=c, eps=eps)
    dat.Enoncouple = Enoncouplefunc(par)
    E2, V2 = BlocksED(par)
    dat.E2_list = E2.reshape(par.Nsite, 2)
    dat.V2_list = V2.reshape(par.Nsite, 2, 2)
    dat.dH0 = dat.omega**2 * dat.R
    dat.F1 = Force(par, dat.dH0, dat.ci, dat.c)
    return dat


# ---------------------------------------------------------------------------
# ORIGINAL function retained: runTraj
# ---------------------------------------------------------------------------
def runTraj(parameters, case, psi_initial, seed):
    """One trajectory; main run.py handles phases, cases, and NTraj averaging.

    Same original loop structure: initialize dat, save populations at nskip,
    call VelVer(dat).  Returns the original suite's 2D-compatible dictionary.
    """
    dat = create_dat(parameters, case, psi_initial, seed)
    NSteps = parameters.NSteps
    nskip = parameters.nskip
    Nsite = parameters.Nsite
    step_idx = np.arange(0, NSteps+1, nskip, dtype=int)
    if step_idx[-1] != NSteps:
        step_idx = np.r_[step_idx, NSteps]

    rhoX = np.empty((len(step_idx), parameters.Nx, parameters.Ny), dtype=float)
    rhoC = np.empty_like(rhoX)
    norms = np.empty(len(step_idx))
    energies = np.empty(len(step_idx))

    # Original 1D rho_ensemble used a flat, mixed real/k space array.
    # Our ci is full real space; compute both density maps directly.
    for k, s in enumerate(step_idx):
        if k:
            for _ in range(int(s - step_idx[k-1])):
                dat = VelVer(dat)
        rhoX[k] = np.abs(dat.ci[0])**2
        rhoC[k] = np.abs(dat.ci[1])**2
        norms[k] = total_norm(dat.ci)
        energies[k] = energy(dat)

    return {'steps':step_idx, 'times_au':step_idx*parameters.dtN,
            'rhoX':rhoX, 'rhoC':rhoC, 'norm':norms, 'energy':energies}
