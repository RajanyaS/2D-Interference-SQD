# SQD-style `serial.py` / `runjobs.py` for the **2D two-material MFE interferometer**

This adaptation keeps the original workflow

```
runjobs.py -> prepare scan folders -> sbatch serial.py -> avg.py -> plot_results.py
```

while matching the new `run.py`'s single-excitation, coherent two-packet, two-material, multimode-Brownian MFE calculation.

## Why the uploaded 1D scripts needed changes

| Uploaded 1D SQD scripts | This 2D adaptation |
|---|---|
| `serial.py` calls `method.runTraj(par)` | `serial.py` calls `run.main(input,output,traj_start,traj_count)` to preserve same/different and φ=0/π loops |
| `np.savetxt` outputs two 1D density arrays | Save time-resolved 2D `rhoX`, `rhoC`, `norm`, `energy` in compressed `.npz` per material/phase |
| `runjobs.py` scans `Nex` and single γ, ω | Scan `brownian_amplitude_multiplier_2` or any editable 2D model parameter; `Nex` is fixed to one quantum excitation |
| Copies only 1D modules | Copies `Model`, `Method`, `spectra` and plotting, run, batch, and aggregation scripts |
| No batch ID in thermal seeds | Each job gets a **nonoverlapping** range `itraj = traj_start ... traj_start + traj_count - 1` |
| Mean of equally sized text files | `avg.py` uses **trajectory-count-weighted** means and recomputes interference diagnostics |
| Periodically re-randomized `SEED` | Fixed model `SEED` and `initial_seed`; same trajectory index shares thermal random draws across cases/phases |

No change to the Hamiltonian, `Model/polaritonmod.py`, `initialMC.py`, or `Method/split_op_Saeed.py` was needed for cluster parallelization. The adapted `run.py` accepts two additional optional CLI arguments, but **calling it with no new arguments is unchanged**.

## Cluster example: 100 thermal trajectories for each case/phase

Run commands **from the project root**. This creates **5 Slurm jobs × 20 independent thermal trajectories**. Each trajectory index is used for all four conditions (`same` and `different`, φ=0 and π), keeping a paired comparison.

```bash
python runjobs.py \
    --input input_brownian.txt \
    --output-root cluster_runs \
    --jobs 5 \
    --traj-per-job 20 \
    --mode prepare
```

Review `cluster_runs/default/input.txt`; `NTraj` is automatically set to `5*20=100` in the generated file. Once checked, run with the same arguments plus `--reuse --mode sbatch`:

```bash
python runjobs.py \
    --input input_brownian.txt \
    --output-root cluster_runs \
    --jobs 5 \
    --traj-per-job 20 \
    --reuse \
    --mode sbatch
```

`serial.py` uses a portable `#!/usr/bin/env python3` shebang and includes example `#SBATCH` resource directives. Ensure the cluster job environment makes the correct Python/NumPy version available. On the cluster shown in the original SQD example, you may replace the shebang by the site-specific Anaconda Python path, if needed.

Monitor jobs via `squeue -u "$USER"`. Slurm logs are in `cluster_runs/default/logs/`. All batches should finish successfully before aggregation.

```bash
cd cluster_runs/default
python avg.py --input input.txt --root . --output results_ensemble
python plot_results.py --input input.txt --results results_ensemble
python animate_interference.py --results results_ensemble \
    --output results_ensemble/interference_evolution.gif
```

The outputs `results_ensemble/same_interference.npz`, `different_interference.npz`, and `different_minus_same.npz` have the **same structure** as an unsplit call to `run.py`.

## Example material-parameter scan

Create 2 folders, one with identical material baths and another with 1.4×-scaled material-2 bath:

```bash
python runjobs.py --input input_brownian.txt --output-root J_scan \
    --jobs 5 --traj-per-job 20 \
    --scan brownian_amplitude_multiplier_2=1.0,1.4 \
    --mode sbatch
```

Remember: the factor 1.4 is an **illustration**, not a separate fitted material. Both spectral densities use the supplied six-Brownian formula and **no `nsites` divisor**. You can scan `eps2_ev` or other parameters using additional `--scan` flags.

## Quick local debug

```bash
python runjobs.py --input input_brownian_smoke.txt --output-root debug_jobs \
    --jobs 2 --traj-per-job 2 --mode local
```

This automatically runs the two jobs, merges the four independent thermal samples per case/phase, and plots. For real simulations, increase the simulated time and trajectory count after checking the arrival and overlap of the wavepackets.

## One-job manual route (closest to old `serial.py`)

```bash
python serial.py --input input_brownian.txt --output single_job --traj-start 0 --traj-count 100
python plot_results.py --input input_brownian.txt --results single_job
```

This produces standard results without requiring `runjobs.py` or `avg.py`.

## Numerical validation performed

* Existing **22** physics and spectral tests pass.
* Tested 2 separate batches of 2 thermal samples against a direct 4-sample `run.py`: max density difference was around machine precision; phase-map difference under `3e-20`.
* Tested parameter-scan folder creation for material-2 spectral amplitude 1.0 and 1.4.
* Did not execute actual `sbatch` submission (there is no Slurm scheduler in this environment); the command construction and local execution are verified.

## Scientific caveats

* `NTraj=100` means 100 thermal initial conditions **per case and per phase** (each phase is an independent MFE propagation, paired by initial thermal random samples). Four conditions = 400 propagated MFE trajectories.
* The current result files contain ensemble means, not per-trajectory variances or confidence intervals; they do not by themselves establish statistically significant material differences.
* The standard plotter shows phase-cycled photon-density contrast, not necessarily the exact linear interference term when self-consistent nuclei respond differently in the phase 0 and π simulations.
* Check grid convergence, actual packet collision time, and `NSteps` before interpreting the plots.

## Fix: `ModuleNotFoundError: No module named 'run'` under SLURM

With `sbatch`, SLURM copies `serial.py` to a spool path like
`/var/spool/slurmd/job12345/slurm_script`. Python can then omit the project's
working directory from `sys.path` even though `--chdir` changes the working directory.
The repaired `serial.py` explicitly finds the project directory and prepends it to
`sys.path` **before** importing `run`. It also keeps relative input/output paths
rooted at that project directory. No changes to the physics or seed ranges.

If you have already prepared the jobs with the older `serial.py`, copy the repaired
script **into the prepared case directory**. Updating only the top-level `serial.py`
is insufficient because `runjobs.py --reuse` deliberately reuses existing copies:

```bash
# Execute from the main project root
cp serial.py cluster_runs/default/serial.py
ls -l cluster_runs/default/{run.py,serial.py,input.txt}

# Fast import check, deliberately emulating SLURM's separate script path
cd cluster_runs/default
python -c 'import run; print(run.__file__)'

# Submit again, from the project root (cd ../..)
cd ../..
python runjobs.py --input input_brownian.txt --output-root cluster_runs \
    --jobs 5 --traj-per-job 20 --reuse --mode sbatch
```

If you have multiple scan subfolders, copy the updated serial into **each** scan
folder (`cluster_runs/<scan>/serial.py`). A complete new project download contains
the fix and prepares correctly from a fresh output root.

Inspect `sacct` and `logs/` after submission; `squeue` shows only running/pending jobs.
