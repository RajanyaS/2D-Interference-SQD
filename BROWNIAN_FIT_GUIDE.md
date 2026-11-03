# Using the six fitted Brownian spectral-density components

This is a fully compatible extension of the user's two-material, 2D, one-excitation SQD/MFE project. **Only `Model/polaritonmod.py` needed modification.** The Saeed-style `Method/split_op_Saeed.py`, `initialMC.py`, and the existing runner remain unchanged.

## 1. Exact formula supplied

For each material `s=1,2`:

```python
def JL(w, Gamma, Omega, a):
    denom = ((w + Omega)**2 + Gamma**2) * ((w - Omega)**2 + Gamma**2)
    return a*w/denom

J_s(w) = sum(JL(w, Gamma[s,r], Omega[s,r], a[s,r]) for r in range(6))
a[s,r] = amp_base[s,r] * (1e-11 * pi**3) * multiplier[s]
```

The six `Omega`, six `Gamma` and six `amp_base` numbers for **material 1** are included in `parameters` as the exact values the user supplied. This is a sum of **six** Brownian terms, notwithstanding the fitting comment saying "3 brownians". It may represent three pairs in the underlying fit, but the executable formula contains six terms.

The `Omega`, `Gamma` and `w` are assumed to be **Hartree / atomic angular-frequency units**, so `1 Ha = 219474.63137 cm^-1` in energy-equivalent spectroscopic units. Verify those units against the actual fitting script. This code uses `J` in Hartree; different fitted spectral-density conventions (e.g. correlation function without the pi/2 factor) require rescaling before they can be mapped to the local Holstein bath.

**Normalization choice:** As requested, the Brownian amplitudes are used directly with `1e-11*pi**3` and no `nsites` divisor. The 2D lattice size never rescales the fitted spectrum. The physical units and fit convention should still be validated.

## 2. Map to a discrete local harmonic bath

The SQD MFE Hamiltonian at site `n` of material `s` uses mass-weighted vibrational coordinates:

\[
H_{ph+Xph} = \sum_{n,j}\left[\frac{P_{nj}^2}{2}+\frac{\omega_{sj}^2Q_{nj}^2}{2} + c_{sj} Q_{nj}|X_n|^2\right].
\]

The spectral density convention is

\[
J_s(\omega)=\frac\pi2\sum_j\frac{c_{sj}^2}{\omega_{sj}}\delta(\omega-\omega_{sj}).
\]

Therefore, for a narrow frequency interval `bin_j`, the discretization is

\[
c_{sj}^2 = \frac{2\omega_{sj}}\pi \int_{\mathrm{bin}_j} J_s(\omega)\,d\omega
\quad\longrightarrow\quad
c_{sj}=\sqrt{\frac{2\omega_{sj}}\pi J_s(\omega_{sj})\,\Delta\omega_j}\]

for sufficiently narrow bins. `Model/polaritonmod.py` evaluates this bin integral using **48-point Gauss-Legendre quadrature** to avoid missing narrow high-frequency peaks. The returned arrays are `freq1, J1, coupl1` and `freq2, J2, coupl2`.

`site_bath(case)` broadcasts them across all lattice sites, using material 1 for `x < Nx/2`, material 2 for the other half. The *same* values are reused at every site of one material but each site has **independently sampled local coordinates** `R[x,y,j], P[x,y,j]`.

`initR()` samples harmonic thermal Wigner distributions using the local bath frequency. `Force()` in the existing Saeed method then calculates

\[
F_{n,j} = -\omega_{sj}^2R_{nj} - c_{sj}|X_n|^2.
\]

`propagateCi()` uses `sum_j c_{sj} R_{nj}` as an exciton local-energy shift. **No changes to `Method/split_op_Saeed.py` are needed.**

## 3. How to run

```bash
python plot_brownian_fit.py --input input_brownian.txt
python initialMC.py --input input_brownian.txt
python run.py --input input_brownian.txt --output results_brownian
python plot_results.py --input input_brownian.txt --results results_brownian
python -m unittest discover -s tests -v
```

The included `input_brownian_smoke.txt` is a much smaller end-to-end validation input.

## 4. Material 2: what is physically necessary?

Only **one fitted parameter set** was supplied. The example therefore uses the supplied spectrum for both materials with an **illustrative 1.4× spectral amplitude multiplier on material 2**. This is a controlled test, not evidence that the materials are actually different. To make both materials identical, use `brownian_amplitude_multiplier_2=1.0` and `eps2_ev=eps1_ev`.

When actual second-material fit parameters become available, add in `input_brownian.txt`:

```ini
$brownian_omega2_au = [ ...6 fitted numbers... ]
$brownian_gamma2_au = [ ...6 fitted numbers... ]
$brownian_amp2_base = [ ...6 fitted numbers... ]
$brownian_amplitude_multiplier_2 = 1.0
```

No changes to the MFE propagation are necessary. With `cases = 'same,different'`, `same` uses bath 1 on both sides; `different` uses bath 1 on the left and bath 2 on the right.

## 5. Choosing frequencies and resolution

The six supplied resonance frequencies correspond to approximately **90, 1543, 281, 1723, 1295, and 1207 cm^-1**. The narrowest fitted widths are around **37 cm^-1**. Twelve modes on 70–1800 cm^-1 would have ~157 cm^-1 separation and would poorly resolve these peaks. The provided demonstration uses **64 modes, 30–2100 cm^-1**, with bin-integrated weights. Increase to 128 and 256 modes, and enlarge the frequency window, to check convergence of the interference observables and reorganization energies.

With the unscaled fitted amplitudes, the **64-mode truncated discrete reorganization energy** is about **370.86 cm^-1** for material 1 (and 519.20 cm^-1 for an artificial 1.4×-amplitude material 2). These values depend on the supplied amplitude prefactor and spectral-density convention; verify those against the fit before interpreting them as physical reorganization energies.

One can compare `lambda = sum_j c_j**2/(2*omega_j**2)` across discretizations. Because the integration above preserves *spectral area* per bin, it does not automatically preserve the `J(w)/w`-weighted reorganization energy exactly on coarse grids.

## 6. Important physics caveat

The fitted widths `Gamma_r` shape the **continuum spectral density**. They do not produce explicit friction on the classical oscillators. After discretization, the MFE solver evolves **independent undamped harmonic oscillators**, so the finite discrete bath only approximates a continuous dissipative environment; recurrences can occur. To model the original overdamped/underdamped Brownian correlation function quantitatively, converge the number of modes and consider whether a damped oscillator hierarchy, stochastic bath, or other treatment is needed.
