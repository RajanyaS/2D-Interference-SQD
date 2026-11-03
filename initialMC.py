
import argparse
from pathlib import Path
import numpy as np
from Model.polaritonmod import parameters, EV


def Epol(par, eps_ev=None):
    "Reference/local homogeneous LP dispersion and Hopfield vector (X,C)."
    epsi = (par.eps_k if eps_ev is None
            else par.eps_k + (eps_ev - par.reference_eps_ev)*EV)
    H = np.zeros((par.Nx, par.Ny, 2, 2), dtype=float)
    H[..., 0, 0] = epsi
    H[..., 1, 1] = par.omega_k
    H[..., 0, 1] = H[..., 1, 0] = par.g_k
    eigs, vectors = np.linalg.eigh(H)
    lp = vectors[..., :, 0]
    return eigs[..., 0], lp[..., 0], lp[..., 1]


def _setup(par, side):
    if side not in ('left', 'right'):
        raise ValueError('side must be left or right')
    sign = 1 if side == 'left' else -1
    if par.initial_bulk_mode == 'local':
        eps = par.eps1_ev if side == 'left' else par.eps2_ev
    elif par.initial_bulk_mode == 'reference':
        eps = None
    else:
        raise ValueError('initial_bulk_mode must be reference or local')
    energy, u, v = Epol(par, eps_ev=eps)
    qidx = par.kx_mode % par.Nx
    kyidx = par.ky_mode % par.Ny
    k0 = sign*abs(par.kx[qidx])
    ky0 = par.ky[kyidx]
    x0 = -sign*par.offset_sites*par.a_bohr
    y0 = 0.0
    qxidx = qidx if sign > 0 else (-qidx) % par.Nx
    E0 = (par.E0_ev*EV if par.E0_ev is not None
          else energy[qxidx, kyidx])
    allowed = np.broadcast_to((sign*par.kx[:, None] > 0), energy.shape).copy()
    if par.dE_ev > 0:
        allowed &= np.abs(energy-E0) < par.dE_ev*EV/2
    if not np.any(allowed):
        raise ValueError(f'No allowed LP momenta for {side}; broaden dE_ev or change kx_mode')
    return energy, u, v, allowed, x0, y0, k0, ky0


def _distances2(par, x0, y0):
    """Squared 2D distance to desired packet center, respecting FFT periodicity."""
    Lx, Ly = par.Nx*par.a_bohr, par.Ny*par.a_bohr
    dx = (par.x-x0+Lx/2) % Lx - Lx/2
    dy = (par.y-y0+Ly/2) % Ly - Ly/2
    return dx[:, None]**2 + dy[None, :]**2


def _LP_fields(ak, u, v):
    return (np.fft.ifft2(ak*u, norm='ortho'),
            np.fft.ifft2(ak*v, norm='ortho'))


def _spread(ak, u, v, distances2, metric):
    """Return the weighted mean squared distance of the LP packet."""
    Xr, Cr = _LP_fields(ak, u, v)
    rho = abs(Xr)**2 if metric == 'exciton' else abs(Xr)**2 + abs(Cr)**2
    return float(np.sum(distances2*rho)/np.sum(rho))


def _greedy_mc(ak, allowed, u, v, distances2, par, seed):
    """Greedy Monte Carlo search for a_k that minimizes the LP packet spread."""
    idx = np.flatnonzero(allowed.ravel())
    rng = np.random.default_rng(seed)
    metric = par.mc_objective
    if metric not in ('exciton', 'total'):
        raise ValueError("mc_objective must be 'exciton' or 'total'")
    best = _spread(ak, u, v, distances2, metric)
    start = best
    count = 0
    # Only allowed elements are perturbed. In particular, NEVER symmetrize
    # a(k) with a(-k) as in the original 1D script: the packets must move inward.
    flat = ak.ravel().copy()
    for _ in range(par.ntry):
        index = int(rng.choice(idx))
        candidate = flat.copy()
        angle = rng.uniform(0.0, 2*np.pi)
        amplitude = par.mc_step*(rng.random()-0.5)
        candidate[index] += amplitude*np.exp(1j*angle)
        candidate /= np.linalg.norm(candidate)
        loss = _spread(candidate.reshape(ak.shape),u,v,distances2,metric)
        if loss < best:
            flat = candidate
            best = loss
            count += 1
    return flat.reshape(ak.shape), dict(initial_spread_bohr2=start,
                                       final_spread_bohr2=best,
                                       accepted=count,
                                       ntry=par.ntry,
                                       n_allowed=len(idx))


def sqd_mc_coefficients(par, side, seed):
    """Return the optimized LP coefficients and convergence information.

    This is the same mathematical workflow as original initElectronicOPT:
    1) energy-shell mask, 2) normalized a_k on selected modes,
    3) trial a_k -> a_k + A exp(i theta) on ONE selected mode,
    4) renormalize, 5) LP reconstruction, 6) accept iff spread decreases.

    We replace its original implicit center with separate specified 2D centers,
    and use a phase-translated equal-amplitude seed, not a Gaussian seed.
    """
    if par.dE_ev <= 0:
        raise ValueError("SQD MC localization requires dE_ev > 0 to select an LP energy shell")
    energy,u,v,allowed,x0,y0,k0,ky0 = _setup(par,side)
    KX,KY = np.meshgrid(par.kx,par.ky,indexing='ij')
    # np.fft.ifft2 returns fields at array indices. Convert physical x0,y0 to
    # displacements relative to indices (0,0), NOT relative to coordinate 0.
    k_phase = KX*(x0-par.x[0])+KY*(y0-par.y[0])
    ak = np.where(allowed,np.exp(-1j*k_phase),0).astype(complex)
    ak /= np.linalg.norm(ak)
    distances2 = _distances2(par,x0,y0)
    ak,info = _greedy_mc(ak,allowed,u,v,distances2,par,seed)
    info.update(center_x_bohr=x0,center_y_bohr=y0,center_kx=k0,
                allowed_energy_min_ev=float(np.min(energy[allowed])/EV),
                allowed_energy_max_ev=float(np.max(energy[allowed])/EV))
    return ak,u,v,allowed,info


def packet(par, side, seed):
    """Prepare one incoming LP arm in the common 2D Hilbert space."""
    method = par.initial_method.lower()
    if method == 'sqd_mc':
        ak,u,v,mask,info = sqd_mc_coefficients(par,side,seed)
    elif method == 'gaussian':
        _,u,v,mask,x0,y0,k0,ky0 = _setup(par,side)
        X,Y = np.meshgrid(par.x,par.y,indexing='ij')
        gaussian = np.exp(-0.25*((X-x0)/(par.sigma_x_sites*par.a_bohr))**2
                          -0.25*((Y-y0)/(par.sigma_y_sites*par.a_bohr))**2)
        target = gaussian*np.exp(1j*(k0*(X-x0)+ky0*(Y-y0)))
        ak = np.where(mask,np.fft.fft2(target,norm='ortho'),0.0)
        if np.linalg.norm(ak) < 1e-15:
            raise ValueError(f'Gaussian arm {side} annihilated by selected modes')
        ak /= np.linalg.norm(ak)
        distances2=_distances2(par,x0,y0)
        ak,info=_greedy_mc(ak,mask,u,v,distances2,par,seed)
    else:
        raise ValueError("initial_method must be 'sqd_mc' or 'gaussian'")
    Xr,Cr = _LP_fields(ak,u,v)
    norm = np.sqrt(np.sum(abs(Xr)**2+abs(Cr)**2))
    if norm < 1e-15: raise ValueError('Failed to normalize LP state')
    return Xr/norm,Cr/norm


def initElectronic(par): 
    "initialize electronic wavepackets for left and right arms"
    XL,CL=packet(par,'left',par.initial_seed)
    XR,CR=packet(par,'right',par.initial_seed+11)
    return XL,CL,XR,CR


def initial_state(XL,CL,XR,CR,phi):
    "Combine left and right wavepackets with a relative phase phi."
    phase=np.exp(1j*phi)
    X=XL+phase*XR
    C=CL+phase*CR
    norm=np.sqrt(np.sum(abs(X)**2+abs(C)**2))
    if norm < 1e-12: raise ValueError('The two waves cancel completely')
    return np.stack([X/norm,C/norm])


def save_packets(par, filename):
    "Save the left and right wavepackets to a compressed .npz file."
    XL,CL,XR,CR=initElectronic(par)
    overlap=np.vdot(XL,XR)+np.vdot(CL,CR)
    Path(filename).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(filename,XL=XL,CL=CL,XR=XR,CR=CR,overlap=overlap)
    print(f'Wrote {filename}; method={par.initial_method}; '
          f'overlap=<L|R>={overlap:.5g}')
    if par.initial_method == 'sqd_mc':
        for side,seed in [('left',par.initial_seed),('right',par.initial_seed+11)]:
            _,_,_,_,info=sqd_mc_coefficients(par,side,seed)
            print(f'{side}: {info["n_allowed"]} allowed LP modes; '
                  f'accepted {info["accepted"]}/{info["ntry"]} moves; '
                  f'spread {info["initial_spread_bohr2"]:.5g}'
                  f' -> {info["final_spread_bohr2"]:.5g} bohr^2')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='input.txt')
    p.add_argument('--output',default='InitialStates/two_incident_packets.npz')
    a=p.parse_args()
    save_packets(parameters.from_file(a.input),a.output)
