# SIX-BROWNIAN FIT SUPPORT

**New:** the Brownian fit amplitudes have **no `nsites` divisor**; this version supports `j_source_1 = "brownian"` and `j_source_2 = "brownian"` using the six fitted Brownian terms supplied by the user. See [BROWNIAN_FIT_GUIDE.md](BROWNIAN_FIT_GUIDE.md) and [input_brownian.txt](input_brownian.txt). The Saeed-style propagation and two-packet initializer are unchanged.

---

# 2D two-material SQD-style polariton interferometer (no molecule)

A standalone NumPy/Matplotlib research **prototype** modeled on your uploaded
`polaritonmod.py`, `initialMC.py`, `split_op_Saeed.py`, `input.txt`, `run.py`.
It uses **one total excitation**, a coherent initial superposition of two
counter-propagating wavepackets, **one common 2D cavity**, and independent
site-local harmonic phonon baths, separately parameterized by `J_1(ω)` and
`J_2(ω)`. No molecule exists in this version.

## Run

```bash
pip install numpy matplotlib
python initialMC.py --input input.txt --output InitialStates/two_incident_packets.npz
python plot_localization.py --input input.txt --output results/sqd_mc_localization.png
python run.py --input input.txt --output results
python plot_results.py --input input.txt --results results
python -m unittest discover -s tests -v
```

The runner constructs the packets itself using `initialMC.initElectronic()` to
ensure it uses the identical initial state in all scenarios. The initializer's
standalone command is provided for plotting/reusing the individual packets.
Outputs are `results/*.npz`, summary diagnostics, and `results/*.png`.
**The two example CSV spectra are synthetic demonstration spectra, NOT actual
measured or fitted spectra. Replace them before drawing material conclusions.**

## Files and SQD correspondence

| File | Responsibility |
|---|---|
| `Model/polaritonmod.py` | Units, 2D site and momentum arrays, cavity dispersion, input parser, J(ω) -> discrete couplings, thermal Wigner phonon sampling, two region masks |
| `initialMC.py` | Original SQD-style LP energy-shell constrained random-walk localization in 2D (or optional Gaussian-target mode), coherent state `(L + exp(iφ) R) / norm` |
| `Method/split_op_Saeed.py` | **Uploaded Saeed-style organization restored:** `Epol`, `Enoncouplefunc`, `Ushuf/Ushufrev`, `BlocksED`, `propagateCi` with `cres1..cres5`, `Force`, `VelVer`, `pop`, `runTraj`; also `Ugamma`/`Uk` convenience helpers |
| `run.py` | Paired `same`/`different` thermal trajectories and φ=0/π, outputs and contrast calculations |
| `plot_results.py` | 2D maps, center-window traces, material difference ΔD, spectral input diagnostics |
| `tests/test_physics.py` | Unitarity, eigenvectors, forces, homogeneous control, norm |



The uploaded 1D `split_op_Saeed.py` partly propagates photon coefficients in
momentum space and excitons in real space. Here **both exciton and photon arrays
are explicitly 2D real-space arrays at rest**, and are both FFT2-transformed for
the exact homogeneous `Uk` step. We avoid the ambiguity about which basis a
wavefunction block lives in.

## Original SQD `initialMC.py` localization, implemented in 2D

**The default initializer now uses `initial_method='sqd_mc'` and DOES NOT start
from a Gaussian target.**  It follows the uploaded original 1D code's actual
`initElectronicOPT` method, with two separately chosen target positions.

The source code selects lower-polariton eigenvectors `(u_k,v_k)` from the
homogeneous 2×2 Hamiltonian. For each incoming arm `s`, the only allowed
momenta obey:

```
  abs(E_LP(k) - E0_s) < dE_ev / 2
  kx > 0  for LEFT,     kx < 0 for RIGHT
```

With `initial_bulk_mode='reference'`, the two arms are constructed using the
same reference Hamiltonian. (The second condition uses `sign(kx)` as a proxy
for the inward group velocity, valid for the current reference dispersion;
for other bands check the sign of `dE_LP/dkx`.)

Initialize normalized *momentum coefficients* on the chosen energy shell,
using Fourier translation phases to center them at `x_L=-offset_sites*a`
versus `x_R=+offset_sites*a`, with `y=0`:

```
  a_s(k) = exp[-i k·(r_s - r_grid_origin)] / sqrt(N_allowed)
               if k is allowed, else 0.
  X_s(r) = IFFT2[u_LP(k)*a_s(k)]
  C_s(r) = IFFT2[v_LP(k)*a_s(k)]
```

The *actual SQD localization objective* minimizes the excitonic spatial
second moment about the packet's specified center, rather than minimizing a
Gaussian-target mismatch:

```
 L_s = sum_r |r-r_s|_periodic² |X_s(r)|² / sum_r |X_s(r)|²
```

`mc_objective='total'` instead weights the exciton+photon density. At every
MC iteration, choose ONE allowed momentum index and propose
`a_k -> a_k + A*exp(i*theta)`, renormalize the complete `a` vector, rebuild
the LP wavepacket, and **accept if and only if** the spread has decreased.
Outside-shell coefficients remain zero. In the original uploaded 1D code,
`a2 = a2 + a2[::-1]` was also applied; this MUST NOT be applied independently
to either 2D incoming arm because pairing opposite momenta destroys the desired
inward direction. The 2D objective uses periodic distances to avoid apparent
localization penalties from the FFT periodic boundary.

The energy shell also limits how tightly the wavepacket can be localized:
`dE_ev` should not be so small that it admits only a handful of momentum modes.
The example `dE_ev=0.14` contains 25 allowed modes per direction with the
example grid/parameters; the exact number depends on the dispersion and grid.
`ntry=2500` runs 2500 proposals **for each arm**, with reproducible independent
seeds. Increase the number of trials and verify the extent, group velocities,
and arrival times before using the resulting packet in a publication.

For direct comparison with the earlier code, the alternative
`initial_method='gaussian'` prepares the original 2D Gaussian-target packet,
then retains its LP components and optional MC refinement. Use the preserved
`input_gaussian.txt` for this baseline (it sets `ntry=0`, `dE_ev=0`).

The **only other procedural correction** is that the acceptance test keeps
the accepted coefficients/fields. The uploaded 1D code sometimes carried a
last *rejected* trial's `psi_ex` into its returned diagnostic.

Run `python plot_localization.py --input input.txt` to see before/after
wavepacket densities, target centers, allowed momentum masks, and optimized
second-moment values.

## Underlying Hamiltonian

`X_n†` creates a host exciton at two-dimensional cell `n`; `a_q†` creates a
cavity photon, with a common cavity across materials 1 and 2. In atomic units,

```
H = sum_q [ eps_ref(q) X_q†X_q + omega_c(q) a_q†a_q
            + g(q)(a_q†X_q + X_q†a_q) ]
  + sum_n [eps_{s(n)} - eps_ref] X_n†X_n
  + sum_{n,j} omega_{s(n),j} b_{n,j}†b_{n,j}
  + sum_{n,j} c_{s(n),j}/sqrt(2 omega_{s(n),j})
                  X_n†X_n (b_{n,j}†+b_{n,j}).
```

The program assumes **zero additional exciton hopping** across the interface.
`tau_ref_cm1=0` by default, as in the uploaded original SQD model. A nonzero
`tau_ref_cm1` adds homogeneous hopping to `eps_ref(q)`; it does not create
separate tau_1, tau_2, tau_12 and should not be interpreted as an interface term.
Polariton propagation across the boundary is still possible through the
**shared dispersive cavity photon**.

```
eps_ref(q)=eps_ref - 2 tau_ref[cos(kx*a)+cos(ky*a)]
omega_c(q) = c_eff sqrt(kx²+ky²+(pi/Lz)²)
g(q) = g * sqrt(omega_0 / omega_c(q))
```

There is exactly **one state with two spatially separated incoming wavepackets**:

```
|Psi_phi(0)> = N_phi ( |Psi_L(0)> + exp(i phi)|Psi_R(0)> ),
state shape (2, Nx, Ny),  sum(|X|² + |C|²) = 1.
```

This is **NOT two excitations**. The finite-φ runs are distinct initial
conditions. Each run propagates one combined wavefunction through the **total**
two-material Hamiltonian, not material 1 for the left arm and material 2 for
the right arm.

## Spectral input: replace synthetic spectra with your fitted J(ω)

Every file is a two-column CSV:

```
# omega_cm1,J_au
0,0
100,0.000123
200,0.000245
...
2000,0.00004
```

Column 1 is **angular frequency expressed as a wavenumber (cm⁻¹)**;
column 2 is **the exciton–phonon spectral density J(ω) expressed in Hartree**
under the convention below. The file must cover the sampled mode grid
`omega_min_cm1` to `omega_max_cm1` and be increasing and nonnegative.
Do NOT pass raw vibrational DOS or arbitrary normalized IR/Raman intensity as
J; molecular/host exciton–phonon coupling weights are required.

```
J_s(w) = (pi/2) sum_j (c_sj²/w_sj) delta(w-w_sj)
c_sj = sqrt( (2 w_sj/pi) * J_s(w_sj) * Delta_w )
lambda_s = sum_j c_sj² / (2 w_sj²)
```

The grid uses a uniform frequency grid of `n_modes` points and a rectangular
quadrature Δω. The displayed `lambda_s` is the **discrete** reorganization
energy, not necessarily the continuous target value. Increase `n_modes` and
check both λ and dynamics for convergence. A spectral density fit with modes
above `omega_max_cm1` is truncated and must be reassessed.

Set `$j_source_1='csv'`, `$spectrum1='path/to/fitted_1.csv'` and likewise
for material 2. Alternatively choose `$j_source_1='drude'` and use `lambda1_cm1`
and `cutoff1_cm1` for an analytic Drude model. Both sources can be mixed.

### Important comparison controls

* `cases='same,different'`, with `eps2_ev=eps1_ev` (default): **spectral-only** comparison, both halves identical vs bath 2 different.
* `cases='same,different'`, with `eps2_ev!=eps1_ev`: *combined* exciton-offset and bath differences; cannot attribute all effects to J.
* `cases='same,spectral_only,energy_only,different'`: isolate bath and exciton-energy differences separately.
* Set identical spectra and same `eps1_ev=eps2_ev` as a null test: both configurations should match trajectory by trajectory for identical random seeds.

In all comparisons, the same prepared incident **reference** state is reused.
To instead prepare approximate bulk LP eigenvectors of the two materials use
`initial_bulk_mode='local'`, but note that this changes the incident state as
well as the material and can confound comparisons. For the cleanest bath-only
control, use `reference`.

## Semiclassical MFE equations and propagation

The mass-weighted classical oscillator coordinates obey

```
i d/dt Psi = H_el[Q(t)] Psi
Q_dot(n,j) = P(n,j)
P_dot(n,j) = -omega_{s(n),j}² Q(n,j) - c_{s(n),j}|X_n|²
```

The `Q,P` arrays have shape `(Nx,Ny,n_modes)`. Thermal initial conditions
are drawn from quantum harmonic Wigner Gaussians (mean 0, variance
`Q²=1/[2w*tanh(beta*w/2)]`, `P²=w/[2*tanh(beta*w/2)]`) and subsequently
evolved classically. The MFE approximation does not model exact quantum bath
entanglement or relaxation.

Electronic Strang step at frozen coordinates:

```
U_e(dt;Q) ≈ U_local(dt/2;Q) * U_k(dt) * U_local(dt/2;Q)
```

`Uk` performs **FFT2 for X and C**, exact 2×2 `exp(-iH0(q)*dt)` for every q,
then **IFFT2 for X and C**. The phonon shifts and local material energies
act through `Ugamma` on X in 2D real space.

One **symmetric** MFE step:

```
1. electronic half time dtN/2 at old Q (e_substeps_half substeps)
2. evaluate F from midpoint electronic population
3. nuclear velocity Verlet full dtN with same midpoint electronic population
4. electronic half time dtN/2 at updated Q (same number of substeps)
```

This is a second-order time-symmetric splitting, but it is approximate:
check `dtN` and `e_substeps_half` convergence. There is no electronic loss,
no dephasing Lindblad operator, and no non-Hermitian normalization in the
baseline. Photon/exciton norm should stay exactly 1 up to rounding.

## What to track for interference

At each time and lattice cell, save `I_phi(n,t)=|C_phi(n,t)|²` and the exciton
density `|X_phi(n,t)|²` for separate relative phases `phi=0` and `phi=π`.

**1. Spatial interference map:** `D(n,t)=I_0(n,t)-I_pi(n,t)`.

**2. Center-window signed response:** `D_center(t)=sum_{n in W}D(n,t)`.

**3. Center-window signed normalized contrast:**
`D_center(t)/sum_{n in W}[I_0(n,t)+I_pi(n,t)]`.

**4. Window-wide phase sensitivity:**
`sum_{W} |D(n,t)| / sum_{W}[I_0(n,t)+I_pi(n,t)]`.
This keeps oppositely signed fringes from cancelling in the window. It is
not a textbook fringe visibility.

**5. Bath-induced difference:**
`DeltaD(n,t)=D_different(n,t)-D_same(n,t)`. The same initial electronic state,
relative phases, and thermal random numbers (paired-seed sampling) are used
for fair comparison. Also plot `DeltaD_center(t)`.

**6. Numerical diagnostics:** norm, total energy, photo/exciton populations,
initial arm overlap, and convergence with step size, grid size, n_modes,
number of trajectories. For an actual experiment, cavity outcoupling and
noise must be modeled before identifying the simulation's photon density
with a measured intensity.

In linear, frozen-bath propagation, `D` isolates an interference cross term
for equal-norm orthogonal arms. But **full MFE is nonlinear** because forces
depend on populations: different phase runs drive different nuclear histories.
Therefore `D` is an **operational phase-cycling contrast**, not generally
exactly `2 Re(C_L* C_R)` once the baths back-react. For true phase-optimized
fringe visibility, sweep more than two relative phases and assess noise.

## Boundaries, initialization, and interpretation

* FFT2 imposes **periodic boundary conditions**. Packets may wrap around,
  especially for long trajectories. Use a much larger grid, inspect boundary
  densities, and/or restrict the detection window to times before wraparound.
* Restricting initial momentum to one propagation direction is a bandlimit;
  it creates **nonzero spatial tails** even at t=0. Do not mistake initial
  overlap tails for a collision-induced interference signal. Compare packet
  trajectories and require the main lobes to reach the center.
* Material 'same' is homogeneous throughout; 'different' means the *right*
  material receives J2 (and eps2 when different). A different J can alter
  response even if eps1=eps2, without any explicit hopping term.
* Wavepacket preparation within a hard LP energy shell may produce spatial
  tails or broaden packets; inspect the localization diagnostic. With the
  **new default** `initial_method='sqd_mc'`, `$ntry>0` uses the original SQD
  energy-shell constrained greedy-MC procedure, while `$ntry=0` retains only
  its initial equal-amplitude translated energy-shell packet. Gaussian target
  preparation is an explicit separate mode (`input_gaussian.txt`).
* `NTraj=3` in `input.txt` is a **quick demonstration only**. Report uncertainty
  with far more independent thermal trajectories before claiming detection.
* The spectral examples are made-up test curves. Neither the plots nor the
  material differences constitute a validated chemical/material prediction.

## Optional four-phase visibility diagnostic

Set `$phases = '0,pi/2,pi,3pi/2'` and rerun to obtain
`four_phase_center_sampled_visibility(t)` and the first-harmonic response
maps in the `*_interference.npz` outputs. Four-phase sampled visibility uses
`(max_phi I_center - min_phi I_center)/(max_phi I_center + min_phi I_center)`;
the optimum extrema may lie between the four sampled phases. MFE feedback can
also make phase dependence non-sinusoidal, so this is an operational measure,
not automatically a first-order optical coherence function.

## Animated fringe comparison

After generating the ensemble results, run

```bash
python animate_interference.py --results results --output results/interference_evolution.gif
```

The animation displays `D_same(x,y,t)`, `D_different(x,y,t)`, and their
separately rescaled difference `DeltaD`. The two `D` panels share a color
scale, while the much smaller differential signal has its own colorbar.
