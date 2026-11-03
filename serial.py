#!/usr/bin/env python3
#SBATCH --job-name=POL2D_MFE
#SBATCH --time=04:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
# Slurm stdout path is set by runjobs.py to the prepared logs/ directory.
"""SQD-compatible single cluster job for 2D two-material polariton MFE.

Original SQD: serial.py imports Method and calls runTraj(parameters), writing
plain text.  Here run.py coordinates same/different material and phase 0/pi
within EACH thermal seed batch, saving 2D time-resolved NPZ data.

Example:
 python serial.py --input input.txt --output job_000 --traj-start 0 --traj-count 20
 sbatch --output=logs/job_%j.out serial.py --input input.txt --output job_000 --traj-start 0 --traj-count 20
"""
import os
# One BLAS thread per Slurm task, avoids multiple processes oversubscribing.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import sys
import time
from pathlib import Path

# IMPORTANT for SLURM: sbatch copies serial.py to /var/spool/slurmd/.../slurm_script.
# When Python runs that copy, sys.path[0] is the SPOOL directory, not necessarily
# the directory with run.py, Model/, Method/, and initialMC.py.
# runjobs.py sets --chdir to the prepared case directory, so cwd comes first.
# SLURM_SUBMIT_DIR provides a useful fallback for manual sbatch submissions.
_candidates = [Path.cwd()]
if os.environ.get('SLURM_SUBMIT_DIR'):
    _candidates.append(Path(os.environ['SLURM_SUBMIT_DIR']))
_candidates.append(Path(__file__).resolve().parent)  # ordinary non-SLURM Python
project_dir = next(
    (folder for folder in _candidates if (folder / 'run.py').is_file()
     and (folder / 'Model' / 'polaritonmod.py').is_file()), None
)
if project_dir is None:
    raise FileNotFoundError(
        'Could not locate run.py and Model/polaritonmod.py. '
        f'Working directory={Path.cwd()!s}; SLURM_SUBMIT_DIR='
        f'{os.environ.get("SLURM_SUBMIT_DIR", "(unset)")}. '
        'Submit with --chdir pointing to the prepared job directory.'
    )
# Ensure both imports and relative input/output paths resolve in the project.
os.chdir(project_dir)
sys.path.insert(0, str(project_dir))

from run import main as run_2d


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', default='input.txt')
    ap.add_argument('--output', default='job_000')
    ap.add_argument('--traj-start',type=int,default=0)
    ap.add_argument('--traj-count',type=int,default=None)
    args=ap.parse_args()
    print('SQD 2D serial job:', args, flush=True)
    print('Working directory:',Path.cwd(),flush=True)
    t0=time.perf_counter()
    run_2d(args.input,args.output,args.traj_start,args.traj_count)
    print(f'Finished in {time.perf_counter()-t0:.3f} s',flush=True)


if __name__=='__main__':
    main()
