
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

EV = 1.0 / 27.211386245988
CM1 = 4.556335252912e-6
AU_TO_FS = 0.02418884326505
BOHR_TO_NM = 0.0529177210903


def _parse_value(value):
    import ast
    value = value.strip()
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        if value.lower() == 'true': return True
        if value.lower() == 'false': return False
        if value.lower() == 'none': return None
        return value


def load_input(path):
    "load key=value options from a text file, ignoring comments and blank lines"
    options = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line or '=' not in line: continue
        key, val = (x.strip() for x in line.split('=', 1))
        options[key.lstrip('$')] = _parse_value(val)
    return options


@dataclass
class parameters:

    Nx: int = 64
    Ny: int = 24
    a_bohr: float = 75.0
    c_eff: float = 137.036 / 2.4
    Lz_bohr: float = 1000.0 * 1.889726125
    eps1_ev: float = 3.2
    eps2_ev: float = 3.2   # same in spectral-only comparison
    reference_eps_ev: float = 3.2
    g_cm1: float = 2.6 * 1500.0
    tau_ref_cm1: float = 0.0  # homogeneous reference hopping ONLY

    # Nuclear variables / MFE
    temperature_K: float = 300.0
    n_modes: int = 12
    omega_min_cm1: float = 70.0
    omega_max_cm1: float = 1800.0
    j_source_1: str = 'csv'  # csv, drude, or brownian
    j_source_2: str = 'csv'
    spectrum1: str = 'spectra/material1_EXAMPLE.csv'
    spectrum2: str = 'spectra/material2_EXAMPLE.csv'
    lambda1_cm1: float = 120.0  # for drude only
    lambda2_cm1: float = 260.0
    cutoff1_cm1: float = 240.0
    cutoff2_cm1: float = 440.0

    # ------------------------------------------------------------------
    # Six-component fitted Brownian spectral density, as supplied by user.
    # ALL omega/gamma values here are assumed to be angular-frequency
    # energies in Hartree (hbar=1); NOT in cm^-1 or eV.
    #
    # J(w) = sum_r a_r*w / [((w+Omega_r)^2+Gamma_r^2)
    #                        *((w-Omega_r)^2+Gamma_r^2)]
    # a_r = amplitude_base_r*(1e-11*pi**3)*multiplier_r.
    # No division by nsites or the simulation grid size.
    # ------------------------------------------------------------------
    brownian_omega1_au: tuple = (0.00041, 0.00703, 0.00128,
                                 0.00785, 0.0059, 0.0055)
    brownian_gamma1_au: tuple = (0.0003, 0.00017, 0.00030000000000000003,
                                 0.00018, 0.0002, 0.0005)
    brownian_amp1_base: tuple = (0.00021, 0.11383000000000001,
                                 0.00116, 0.023490000000000004,
                                 0.011600000000000001, 1e-05)
    brownian_omega2_au: object = None  # None -> use material-1 frequencies
    brownian_gamma2_au: object = None  # None -> use material-1 widths
    brownian_amp2_base: object = None  # None -> use material-1 amplitudes
    brownian_amplitude_multiplier_1: float = 1.0
    brownian_amplitude_multiplier_2: float = 1.0
    brownian_fit_prefactor: float = 1e-11 * np.pi**3
    # Gauss-Legendre points used to integrate J(w) over each mode bin.
    brownian_bin_quadrature: int = 48

    # Initial wavepacket: TWO incident packets but ONE total excitation
    kx_mode: int = 2
    ky_mode: int = 0
    offset_sites: float = 8.0
    sigma_x_sites: float = 5.0
    sigma_y_sites: float = 4.0
    E0_ev: object = None
    dE_ev: float = 0.14  # SQD MC requires a nonempty LP energy shell
    initial_bulk_mode: str = 'reference'  # or local (changes packet composition)
    initial_method: str = 'sqd_mc'  # SQD energy-shell MC or Gaussian-target
    mc_objective: str = 'total'  # faithful original SQD second-moment objective
    ntry: int = 2500  # greedy MC proposals PER wavepacket
    mc_step: float = 0.03
    initial_seed: int = 119

    # Propagation and statistics
    dtN: float = 8.0
    e_substeps_half: int = 2
    NSteps: int = 90
    NTraj: int = 3
    nskip: int = 3
    SEED: int = 20261027
    center_halfwidth_x: int = 4
    center_halfwidth_y: int = 6
    cases: str = 'same,different'
    phases: str = '0,pi'
    assert_norm_tol: float = 1e-8
    # input file relative resolution
    input_root: str = '.'

    # constructed
    Nsite: int = field(init=False)
    x: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)
    kx: np.ndarray = field(init=False, repr=False)
    ky: np.ndarray = field(init=False, repr=False)
    omega_k: np.ndarray = field(init=False, repr=False)
    eps_k: np.ndarray = field(init=False, repr=False)
    g_k: np.ndarray = field(init=False, repr=False)
    E_LP: np.ndarray = field(init=False, repr=False)
    material1: np.ndarray = field(init=False, repr=False)
    freq1: np.ndarray = field(init=False, repr=False)
    freq2: np.ndarray = field(init=False, repr=False)
    coupl1: np.ndarray = field(init=False, repr=False)
    coupl2: np.ndarray = field(init=False, repr=False)
    J1: np.ndarray = field(init=False, repr=False)
    J2: np.ndarray = field(init=False, repr=False)
    reorg1: float = field(init=False)
    reorg2: float = field(init=False)

    def __post_init__(self):
        # Check for valid parameter values and construct derived arrays.
        if self.Nx % 2 or self.Ny % 2: raise ValueError('Nx and Ny must be even')
        if self.Nx < 16 or self.Ny < 8: raise ValueError('Grid is too small')
        if self.n_modes < 1: raise ValueError('n_modes must be >= 1')
        if self.dtN <= 0 or self.e_substeps_half < 1: raise ValueError('Time steps must be positive')
        self.Nsite = self.Nx * self.Ny
        self.x = (np.arange(self.Nx) - self.Nx//2) * self.a_bohr
        self.y = (np.arange(self.Ny) - self.Ny//2) * self.a_bohr
        self.kx = 2*np.pi*np.fft.fftfreq(self.Nx, d=self.a_bohr)
        self.ky = 2*np.pi*np.fft.fftfreq(self.Ny, d=self.a_bohr)
        KX, KY = np.meshgrid(self.kx, self.ky, indexing='ij')
        kz = np.pi / self.Lz_bohr
        omega0 = self.c_eff * kz
        self.omega_k = self.c_eff*np.sqrt(KX*KX + KY*KY + kz*kz)
        tau = self.tau_ref_cm1*CM1
        self.eps_k = (self.reference_eps_ev*EV - 2*tau*(np.cos(KX*self.a_bohr)+np.cos(KY*self.a_bohr)))
        self.g_k = self.g_cm1*CM1*np.sqrt(omega0/self.omega_k)
        self.E_LP = ((self.eps_k+self.omega_k)/2 -
                     np.sqrt(((self.eps_k-self.omega_k)/2)**2+self.g_k**2))
        self.material1 = (np.indices((self.Nx,self.Ny))[0] < self.Nx//2)
        freqs = np.linspace(self.omega_min_cm1, self.omega_max_cm1, self.n_modes)
        self.freq1, self.J1, self.coupl1 = self._discretize(freqs,'1')
        self.freq2, self.J2, self.coupl2 = self._discretize(freqs,'2')
        self.reorg1 = float(np.sum(self.coupl1**2/(2*self.freq1**2)))
        self.reorg2 = float(np.sum(self.coupl2**2/(2*self.freq2**2)))


    def brownian_spectrum(self, omega_au, side='1'):
        """The fitted amplitude is used directly, without any division by nsites."""
        #fitted J(w) = sum_r a_r*w / [((w+Omega_r)^2+Gamma_r^2)*((w-Omega_r)^2+Gamma_r^2)]
        #brownian_omegaX_au, brownian_gammaX_au, brownian_ampX_base are all 1D arrays of length 6.

        side = str(side)
        if side not in ('1', '2'):
            raise ValueError("Brownian side must be '1' or '2'")
        def fitted(name):
            value = getattr(self, f'brownian_{name}{side}_au', None) if name != 'amp' else None
            if name == 'amp':
                value = getattr(self, f'brownian_amp{side}_base')
                if value is None: value = self.brownian_amp1_base
            elif value is None:
                value = getattr(self, f'brownian_{name}1_au')
            return np.asarray(value, dtype=float)

        omega0 = fitted('omega')
        gamma = fitted('gamma')
        ampbase = fitted('amp')
        if (omega0.ndim != 1 or len(omega0) == 0
                or omega0.shape != gamma.shape or gamma.shape != ampbase.shape):
            raise ValueError('Omega, Gamma, and amp_base must be nonempty arrays of equal length')
        if (np.any(~np.isfinite(omega0)) or np.any(~np.isfinite(gamma))
                or np.any(~np.isfinite(ampbase)) or np.any(omega0 <= 0)
                or np.any(gamma <= 0) or np.any(ampbase < 0)):
            raise ValueError('Brownian Omega and Gamma must be positive; amplitudes nonnegative')
        multiplier = float(getattr(self, f'brownian_amplitude_multiplier_{side}'))
        if multiplier < 0 or not np.isfinite(multiplier):
            raise ValueError('Brownian amplitude multiplier must be finite and nonnegative')
        a = ampbase * self.brownian_fit_prefactor * multiplier
        w = np.asarray(omega_au, dtype=float)
        if np.any(w < 0):
            raise ValueError('J(w) is defined here for w >= 0')
        # Broadcast frequencies against the 6 fitted Brownian components.
        wcol = w[..., None]
        denom = (((wcol + omega0)**2 + gamma**2)
                 * ((wcol - omega0)**2 + gamma**2))
        return np.sum(a * wcol / denom, axis=-1)

    def _discretize(self, freq_cm, side):
        """J(w)=(pi/2) sum c_j^2 / w_j delta(w-w_j), mass-weighted.
        CSV col1: wavenumber in cm^-1; col2: spectral J in Hartree.
        Use bin-midpoint convention on a uniform cm^-1 grid.
        """
        source = getattr(self,f'j_source_{side}').lower()
        freq = np.asarray(freq_cm)*CM1
        delta = (self.omega_max_cm1-self.omega_min_cm1)/(self.n_modes-1) if self.n_modes>1 else (self.omega_max_cm1-self.omega_min_cm1)
        dw = delta*CM1
        if source == 'csv':
            file = Path(self.input_root)/getattr(self,f'spectrum{side}')
            data = np.loadtxt(file,delimiter=',',comments='#',ndmin=2)
            if data.ndim!=2 or data.shape[1]<2 or len(data)<2: raise ValueError('Need 2-column CSV of omega_cm1,J_au')
            x,j = data[:,0],data[:,1]
            if np.any(np.diff(x)<=0) or np.any(j<0): raise ValueError(f'Invalid spectrum {file}')
            if freq_cm[0]<x[0] or freq_cm[-1]>x[-1]:
                raise ValueError(f'Spectrum {file} must cover mode frequency interval')
            J = np.interp(freq_cm,x,j)
        elif source == 'drude':
            lam=getattr(self,f'lambda{side}_cm1')*CM1
            gamma=getattr(self,f'cutoff{side}_cm1')*CM1
            J=2*lam*gamma*freq/(freq*freq+gamma*gamma)
        elif source == 'brownian':
            # Exact same six-Brownian sum as the fit, on each material side.
            J = self.brownian_spectrum(freq, side)
            # Integrate J over each frequency interval to avoid missing
            # narrow high-frequency peaks when n_modes is moderately small.
            # Then c_j^2 = (2*w_j/pi) * integral_bin[J(w) dw].
            # This tends to the original midpoint expression as dw -> 0.
            qn = int(self.brownian_bin_quadrature)
            if qn < 4: raise ValueError('brownian_bin_quadrature must be >= 4')
            from numpy.polynomial.legendre import leggauss
            nodes, weights = leggauss(qn)
            lo = np.maximum(0., freq - dw/2)
            hi = freq + dw/2
            nodes_au = (lo[:, None] + hi[:, None])/2 + (hi-lo)[:, None]/2 * nodes[None, :]
            values = self.brownian_spectrum(nodes_au, side)
            areas = (hi-lo)/2 * np.sum(values * weights[None, :], axis=1)
            coupl = np.sqrt(np.maximum(0., 2*freq*areas/np.pi))
            return freq, J, coupl
        else: raise ValueError(f'Unknown J source: {source}')
        coupl=np.sqrt(2*freq*J*dw/np.pi)
        return freq,J,coupl

    def site_bath(self, case):
        """Return local omega, c arrays; 'same' means physically identical halves.
        Cases 'different' and 'spectral_only' use bath 2; different includes eps2.
        """
        bath2_on = case in ('different','spectral_only')
        w2 = self.freq2 if bath2_on else self.freq1
        c2 = self.coupl2 if bath2_on else self.coupl1
        omega = np.where(self.material1[...,None], self.freq1, w2)
        coupling = np.where(self.material1[...,None], self.coupl1, c2)
        if case == 'different': right_eps = self.eps2_ev
        elif case in ('same','spectral_only'): right_eps = self.eps1_ev
        elif case == 'energy_only':
            right_eps = self.eps2_ev
            omega = np.where(self.material1[...,None], self.freq1, self.freq1)
            coupling = np.where(self.material1[...,None], self.coupl1, self.coupl1)
        else: raise ValueError(f'Unknown case: {case}')
        eps=np.where(self.material1, self.eps1_ev, right_eps)*EV
        return omega,coupling,eps

    @classmethod
    def from_file(cls, fname):
        path = Path(fname).resolve()
        overrides=load_input(path)
        overrides.pop('Model',None)
        overrides.pop('Method',None)
        overrides.pop('System',None)
        overrides.pop('Cpus',None)
        overrides['input_root']=str(path.parent)
        return cls(**overrides)


def initR(par, seednum, case='different'):
    """Thermal Wigner sampling, paired by seed across all comparison cases."""
    rng=np.random.default_rng(seednum)
    omega,coup,eps=par.site_bath(case)
    # beta expressed in inverse Hartree. k_B = 3.166811563e-6 Hartree/K
    beta=1.0/(3.166811563e-6*par.temperature_K)
    vq=1/(2*omega*np.tanh(0.5*beta*omega))
    vp=omega/(2*np.tanh(0.5*beta*omega))
    zq=rng.standard_normal(size=(par.Nx,par.Ny,par.n_modes))
    zp=rng.standard_normal(size=(par.Nx,par.Ny,par.n_modes))
    return np.sqrt(vq)*zq, np.sqrt(vp)*zp


def dHel0(par, R, case):
    #Return the local energy shift due to the bath, for a given case.
    omega, _, _ = par.site_bath(case)
    return omega**2*R
