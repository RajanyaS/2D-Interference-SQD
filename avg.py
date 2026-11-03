#!/usr/bin/env python3
"""Combine independent Slurm trajectory batches into standard 2D results.

Outputs exactly the filenames required by plot_results.py and
animate_interference.py.  Averaging is weighted by the actual number of
independent trajectories in each batch, NOT by the number of job files.

Usage: python avg.py --input input.txt --root . --output results_ensemble
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from Model.polaritonmod import parameters
from run import compute_phase_diagnostics, center_window, parse_phases


def combine(input_file='input.txt', root='.', output='results_ensemble', allow_incomplete=False):
    p=parameters.from_file(input_file)
    base=Path(root)
    dst=Path(output)
    dst.mkdir(parents=True,exist_ok=True)
    jobs=sorted(x.parent for x in base.glob('job_*/summary.json'))
    if not jobs:
        raise FileNotFoundError(f'No job_*/summary.json under {base.resolve()}')

    expected_hash=hashlib.sha256(Path(input_file).read_bytes()).hexdigest()
    coverage=set()
    ranges=[]
    for folder in jobs:
        s=json.loads((folder/'summary.json').read_text())['settings']
        if s.get('input_sha256') != expected_hash:
            raise ValueError(f'Input differs from project input: {folder}')
        if s['Nx']!=p.Nx or s['Ny']!=p.Ny or s['NSteps']!=p.NSteps or not np.isclose(s['dtN_au'],p.dtN):
            raise ValueError(f'Incompatible model/time grid for job {folder}')
        if s['SEED']!=p.SEED:
            raise ValueError(f'Thermal random seed base changed in {folder}')
        start=int(s['traj_start']);count=int(s['traj_count'])
        if count<=0 or start<0:raise ValueError(f'Invalid trajectory range in {folder}')
        new=set(range(start,start+count))
        if coverage.intersection(new):raise ValueError(f'Overlapping trajectory seeds in {folder}')
        coverage.update(new)
        ranges.append((folder,count,start))
    if not allow_incomplete and coverage != set(range(p.NTraj)):
        missing=sorted(set(range(p.NTraj))-coverage)
        extras=sorted(coverage-set(range(p.NTraj)))
        raise ValueError(f'Incomplete scan: expected indices 0..{p.NTraj-1}; '
                         f'missing={missing[:20]}, extras={extras[:20]}; '
                         'use --allow-incomplete if intentional')

    first = ranges[0][0]
    for fname in ('initial_packets.npz','center_mask.npy'):
        # Already common to all trajectory batches because input and initial seeds match.
        (dst/fname).write_bytes((first/fname).read_bytes())
    mask=np.load(dst/'center_mask.npy')
    if not np.array_equal(mask,center_window(p)):
        raise ValueError('Center window mask differs from current input')

    total=sum(count for _,count,_ in ranges)
    wanted_cases=[x.strip() for x in p.cases.split(',') if x.strip()]
    wanted_phases=[label for label,_ in parse_phases(p.phases)]
    output_files={}
    summary={'settings':{'Nx':p.Nx,'Ny':p.Ny,'NTraj':total,
                         'expected_NTraj':p.NTraj,
                         'number_of_jobs':len(jobs),
                         'seed_indices':sorted(coverage),
                         'source_input_sha256':expected_hash}}
    for case in wanted_cases:
        output_files[case]={}
        for label in wanted_phases:
            agg=None
            for folder,n,_ in ranges:
                src=folder/f'{case}_phi_{label}.npz'
                if not src.exists():raise FileNotFoundError(src)
                with np.load(src) as data:
                    if agg is None:
                        agg={'steps':data['steps'].copy(),
                             'times_au':data['times_au'].copy()}
                        for k in ('rhoX','rhoC','norm','energy'):
                            agg[k]=data[k].copy()*(n/total)
                    else:
                        for k in ('steps','times_au'):
                            if not np.array_equal(agg[k],data[k]):
                                raise ValueError(f'Incompatible time grid in {src}')
                        for k in ('rhoX','rhoC','norm','energy'):
                            agg[k] += data[k]*(n/total)
            out=dst/f'{case}_phi_{label}.npz'
            np.savez_compressed(out,**agg)
            output_files[case][label]=str(out)
            summary[f'{case}_{label}']={
                'max_norm_error':float(np.max(np.abs(agg['norm']-1))),
                'max_energy_drift_Hartree':float(np.max(np.abs(agg['energy']-agg['energy'][0]))),
                'total_photon_final':float(agg['rhoC'][-1].sum())}
        compute_phase_diagnostics(output_files[case],mask,dst/f'{case}_interference.npz')

    if 'same' in output_files and 'different' in output_files and all(
        k in output_files['same'] and k in output_files['different'] for k in ('0','pi')
    ):
        with np.load(dst/'same_interference.npz') as a, np.load(dst/'different_interference.npz') as b:
            np.savez_compressed(dst/'different_minus_same.npz',
                times_fs=a['times_fs'],delta_Dmap=b['Dmap']-a['Dmap'],
                delta_center_D=b['center_D']-a['center_D'],
                delta_center_brightness=b['center_brightness']-a['center_brightness'],
                delta_signed_contrast=b['normalized_signed_contrast']-a['normalized_signed_contrast'])
    (dst/'summary.json').write_text(json.dumps(summary,indent=2))
    print(f'Merged {len(jobs)} jobs, {total} trajectories per case/phase into {dst.resolve()}')
    return dst


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',default='input.txt')
    ap.add_argument('--root',default='.')
    ap.add_argument('--output',default='results_ensemble')
    ap.add_argument('--allow-incomplete',action='store_true')
    a=ap.parse_args()
    combine(a.input,a.root,a.output,a.allow_incomplete)
