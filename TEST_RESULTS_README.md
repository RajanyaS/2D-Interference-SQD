# Distinct J1/J2 six-Brownian 2D SQD–MFE numerical preview

This is a **completed, preliminary** 12-trajectory-per-condition numerical test using the two-material 2D SQD code and the user's previously provided `input_brownian_distinct_materials.txt`.

## Exact setup
- `Nx=64`, `Ny=24`; 64 mass-weighted oscillators per lattice site.
- Material 1: user's six-Brownian fit; material 2: synthetic **different** Brownian frequencies, widths, and amplitudes (NOT multiplied 1.4 times J1).
- Discretized reorganization energies J1: 370.8603 cm^-1, J2: 370.2596 cm^-1.
- 300 steps, `dtN=8` atomic units, 58.053 fs; 61 snapshots (skip=5).
- 12 seeded thermal trajectories per phase and case (48 complete trajectories total), using the SAME seeds for matched comparisons.
- Cases: J1/J1 and J1/J2; phases: 0 and pi.
- Input differs from supplied file **only** by `$NTraj = 12` instead of 100 (`input_distinct_preview_12traj.txt`).
- Maximum quantum norm error over all cases: about 6.07e-13.
- All four phase/case ensembles and interferometric .npz files were written successfully. The initial command reached the timeout while writing final summary/comparison; the comparison file and `preview_summary.json` were reconstructed from the successfully written complete arrays without rerunning or altering the physics.

## Saved figures
- `preview_photon_interference_12traj.png`: photon phase difference D for both materials and delta D, at t=24.19 and 43.54 fs.
- `preview_exciton_interference_12traj.png`: same for exciton density.
- `preview_raw_density_maps_12traj.png`: raw phi=0 and pi exciton/photon spatial densities at 24.19 fs.
- `preview_interference_time_traces_12traj.png`: time-dependent central spatial phase sensitivity, central brightness, and material differences.
- `preview_arm_motion_12traj.png`: isolated-arm check using one thermal sample, demonstrating the centers **do not collide** in the 58 fs window.
- `same_vs_different_maps.png`, `center_interference_diagnostics.png` etc.: output of existing `plot_results.py` (its automatic representative time was ~0.97 fs; use the preview figures for later times).

## Critical interpretation
- The prepared packets have ~0.37% photonic population each and are strongly excitonic.
- An isolated-arm diagnostic shows the packet centers move from -7.84,+7.95 sites at t=0 to -5.90,+6.30 sites at t=58 fs, respectively: **their centers do not collide**. The spatial wavepacket tails already overlap significantly at t=0.
- The observed phase-dependent fringes are valid outputs of the coherent superposition calculation, but not evidence of a full head-on collision.
- At 24.19 fs, central S is 0.5077 (J1/J1) vs 0.4949 (J1/J2); at 43.54 fs, S is 0.2625 vs 0.1902. Photon brightness drops from roughly 0.0010 to 0.0003 over these times.
- The 12-trajectory ensemble is **not converged**: no uncertainty intervals, no proof that the late-time material-dependent change is statistically significant. For quantitative science, check uncertainty vs NTraj, timestep/nuclear mode discretization, and tune LP detuning/direction to verify actual packet collision.

## To reproduce a small preview
From the project directory:

```bash
python run.py --input input_distinct_preview_12traj.txt --output results_distinct_preview_12traj
python plot_results.py --input input_distinct_preview_12traj.txt --results results_distinct_preview_12traj
python preview_patterns.py --results results_distinct_preview_12traj
```

Run on a workstation/cluster with sufficient memory and runtime. The full 100-trajectory input is available as `input_brownian_distinct_materials.txt`.
