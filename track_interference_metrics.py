#!/usr/bin/env python3
"""Analyze ensemble-averaged spatial phase sensitivity and brightness.

Run from your simulation directory:
  python track_interference_metrics.py --results results_ensemble

S(t)=sum_W |I_0-I_pi| / sum_W (I_0+I_pi) is NOT a standard
fringe visibility or a direct measurement of first-order coherence.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', default='results_ensemble')
    ap.add_argument('--min-relative-brightness', type=float, default=0.10,
                    help='Require both material cases to have >= this fraction of peak mean brightness')
    args=ap.parse_args()
    directory=Path(args.results)
    records={}
    for name in ('same', 'different'):
        path=directory/f'{name}_interference.npz'
        if not path.is_file():
            raise FileNotFoundError(f'{path} not found; run avg.py first')
        with np.load(path) as data:
            wanted=['times_fs', 'normalized_abs_phase_sensitivity', 'center_brightness']
            missing=[k for k in wanted if k not in data]
            if missing: raise KeyError(f'{path}: missing {missing}')
            records[name]={k:np.asarray(data[k]).copy() for k in wanted}
    a=records['same']; b=records['different']
    t=a['times_fs']
    if not np.allclose(t,b['times_fs']):
        raise ValueError('Time axes differ: cannot compare material cases')
    Ss=a['normalized_abs_phase_sensitivity']
    Sd=b['normalized_abs_phase_sensitivity']
    Bs=a['center_brightness']
    Bd=b['center_brightness']
    delta=Sd-Ss
    avg_brightness=(Bs+Bd)/2
    top=float(np.max(avg_brightness))
    if top<=1e-15: raise ValueError('No central photon signal; check packet arrival and detection window')
    threshold=args.min_relative_brightness * top
    good=(Bs>=threshold)&(Bd>=threshold)
    if not np.any(good):
        print('WARNING: No frames where both material cases meet the brightness threshold')
        print('Try --min-relative-brightness 0.01, or check initial packet propagation')
    ti=int(np.argmax(avg_brightness))
    print(f'Maximum mean central photon brightness: t={t[ti]:.4f} fs')
    print(f'  same:      S={Ss[ti]:.6g}, B={Bs[ti]:.6g}')
    print(f'  different: S={Sd[ti]:.6g}, B={Bd[ti]:.6g}')
    print(f'  delta S = different - same = {delta[ti]:+.6g}')
    if np.any(good):
        valid=np.flatnonzero(good)
        ii=int(valid[np.argmax(np.abs(delta[good]))])
        print(f'Max |delta S| at sufficiently bright frames: t={t[ii]:.4f} fs')
        print(f'  delta S={delta[ii]:+.6g}, same S={Ss[ii]:.6g}, different S={Sd[ii]:.6g}')
    fig,axs=plt.subplots(3,1,figsize=(9,9),sharex=True,layout='constrained')
    axs[0].plot(t,Ss,label='same J')
    axs[0].plot(t,Sd,label='different J')
    axs[0].set_ylabel('Spatial phase sensitivity S')
    axs[0].set_ylim(bottom=0)
    axs[0].legend()
    axs[1].plot(t,Bs,label='same J')
    axs[1].plot(t,Bd,label='different J')
    axs[1].axhline(threshold,linestyle=':',label='brightness threshold')
    axs[1].set_ylabel('Central photon brightness B')
    axs[1].legend()
    axs[2].plot(t,delta,label='S(different) - S(same)')
    axs[2].axhline(0,linestyle=':',linewidth=.8)
    if np.any(good):
        axs[2].scatter(t[good],delta[good],s=12,label='both cases bright')
    axs[2].set_ylabel('Material effect: ΔS')
    axs[2].set_xlabel('Time (fs)')
    axs[2].legend()
    output=directory/'interference_metric_tracking.png'
    fig.savefig(output,dpi=180)
    plt.close(fig)
    print('Saved',output)
    print('NOTE: S is a normalized phase-conditioned spatial density difference, not coherence g^(1).')

if __name__=='__main__':
    main()
