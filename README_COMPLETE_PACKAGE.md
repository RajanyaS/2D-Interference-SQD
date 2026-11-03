# Complete two-material 2D SQD–MFE codebase — distinct Brownian spectra

This consolidated package combines the validated 2D SQD MFE simulation code, original-Saeed-style split operator, two-material fitted Brownian bath model, packet initializer, the SLURM job-splitting / averaging workflow, regression tests, and the **completed preliminary 12-trajectory numerical preview**.

## Important inputs
- `input_brownian_distinct_materials.txt`: full 100-trajectory setup for **different shaped** J1 / J2 Brownian spectra (material 1 user fit; material 2 synthetic illustrative distinct spectrum).
- `input_distinct_preview_12traj.txt`: the previously executed preview configuration (12 trajectories per phase/case, 300 steps).
- `input_brownian.txt`: earlier spectrum-amplitude demonstration and other baseline inputs.

## Local run
```bash
python -m unittest discover -s tests -v
python plot_brownian_fit.py --input input_brownian_distinct_materials.txt
python run.py --input input_distinct_preview_12traj.txt --output results_new_preview
python plot_results.py --input input_distinct_preview_12traj.txt --results results_new_preview
python preview_patterns.py --results results_new_preview
```

## SLURM 100-trajectory ensemble
```bash
python runjobs.py --input input_brownian_distinct_materials.txt --output-root distinct_spectra_runs --jobs 5 --traj-per-job 20 --mode prepare
python runjobs.py --input input_brownian_distinct_materials.txt --output-root distinct_spectra_runs --jobs 5 --traj-per-job 20 --reuse --mode sbatch
squeue -u "$USER"
# After *all* jobs finish successfully:
cd distinct_spectra_runs/default
python avg.py --input input.txt --root . --output results_ensemble
python plot_results.py --input input.txt --results results_ensemble
python animate_interference.py --results results_ensemble --output results_ensemble/interference_evolution.gif
```

**SLURM caveat:** `serial.py` is the corrected version that inserts the working directory into `sys.path`, avoiding `ModuleNotFoundError: No module named run` from the SLURM spool directory. Check environment and SLURM resource settings for your own cluster.

## Existing preview output
`results_distinct_preview_12traj/` contains the 12-trajectory results and publication-style diagnostic figures from the preliminary test. This preview is **not statistically converged**, and the packet centers do **not** fully collide during the 58 fs time window. See `TEST_RESULTS_README.md`.

## Dependencies
Python 3; `numpy`, `matplotlib`, `scipy` (for any scripts that import it). Check imports/versions supported by your cluster's Anaconda environment.
