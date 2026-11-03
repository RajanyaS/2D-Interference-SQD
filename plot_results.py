"""Publication-oriented analysis of phase-cycled 2D interference.
Plots are quantitative diagnostics, not validated molecule/material signatures.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Model.polaritonmod import parameters, CM1


def visualize(outfolder='results',input_path='input.txt'):
    out=Path(outfolder)
    p=parameters.from_file(input_path)
    cased={}
    for name in ('same','different'):
        fname=out/f'{name}_interference.npz'
        if fname.exists():
            cased[name]=np.load(fname)
    if not cased: raise FileNotFoundError('No *_interference.npz files; run run.py first')
    x=np.arange(p.Nx)-p.Nx//2
    y=np.arange(p.Ny)-p.Ny//2
    extent=(y[0]-.5,y[-1]+.5,x[-1]+.5,x[0]-.5)
    def heat(ax,d,title,vlim=None,diverging=False):
        args=dict(aspect='auto',interpolation='nearest',origin='upper',extent=extent)
        if diverging:
            vmax=max(np.max(np.abs(d)),1e-15) if vlim is None else vlim
            im=ax.imshow(d,cmap='RdBu_r',vmin=-vmax,vmax=vmax,**args)
        else:
            im=ax.imshow(d,**args)
        ax.axhline(-.5,ls='--',lw=.8,alpha=.7)
        ax.set_xlabel('y index relative to center')
        ax.set_ylabel('x index relative to center')
        ax.set_title(title,fontsize=10)
        plt.colorbar(im,ax=ax,shrink=.75)

    packets=np.load(out/'initial_packets.npz')
    XL,CL,XR,CR=(packets[z] for z in ('XL','CL','XR','CR'))
    fig,axs=plt.subplots(1,3,figsize=(13.5,4),constrained_layout=True)
    heat(axs[0],abs(XL)**2+abs(CL)**2,'Initial left arm: |X_L|²+|C_L|²')
    heat(axs[1],abs(XR)**2+abs(CR)**2,'Initial right arm: |X_R|²+|C_R|²')
    amp=XL+XR;aph=CL+CR
    norm=float(np.sum(abs(amp)**2+abs(aph)**2))
    heat(axs[2],abs(aph)**2/norm,'Coherent superposition photon density, φ=0')
    fig.suptitle('Two prepared wavepackets; one total excitation',fontsize=12)
    fig.savefig(out/'initial_two_arms.png',dpi=170)
    plt.close(fig)

    # Time frame at maximum total central brightness in first 60% of run:
    case=next(iter(cased))
    d0=cased[case]
    allowed=max(2,int(.6*len(d0['times_fs'])))
    ii=int(np.argmax(d0['center_brightness'][:allowed]))
    for name,meta in cased.items():
        a=np.load(out/f'{name}_phi_0.npz')
        b=np.load(out/f'{name}_phi_pi.npz')
        t=float(meta['times_fs'][ii])
        fig,axs=plt.subplots(1,3,figsize=(13.5,4),constrained_layout=True)
        heat(axs[0],a['rhoC'][ii],f'{name}: photon density, φ=0')
        heat(axs[1],b['rhoC'][ii],f'{name}: photon density, φ=π')
        heat(axs[2],meta['Dmap'][ii],f'Phase difference D, t={t:.2f} fs',diverging=True)
        fig.suptitle('One total excitation: 2D photon interference',fontsize=12)
        fig.savefig(out/f'{name}_interference_map.png',dpi=160)
        plt.close(fig)

    if 'same' in cased and 'different' in cased:
        a,b=cased['same'],cased['different']
        ds=a['Dmap'][ii];dd=b['Dmap'][ii];delta=dd-ds
        common=max(np.max(np.abs(ds)),np.max(np.abs(dd)),1e-15)
        fig,axs=plt.subplots(1,3,figsize=(13.5,4),constrained_layout=True)
        heat(axs[0],ds,'Same materials: D',vlim=common,diverging=True)
        heat(axs[1],dd,'Different J: D',vlim=common,diverging=True)
        heat(axs[2],delta,'Different minus same: ΔD',diverging=True)
        fig.suptitle(f'Bath-induced change of phase contrast, t={a["times_fs"][ii]:.2f} fs',fontsize=12)
        fig.savefig(out/'same_vs_different_maps.png',dpi=170)
        plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(11.5,7.5),constrained_layout=True)
    for name,d in cased.items():
        t=d['times_fs'];axs[0,0].plot(t,d['center_I_0'],label=f'{name}, φ=0')
        axs[0,0].plot(t,d['center_I_pi'],ls='--',label=f'{name}, φ=π')
        axs[0,1].plot(t,d['center_D'],label=name)
        axs[1,0].plot(t,d['normalized_signed_contrast'],label=name)
        axs[1,1].plot(t,d['normalized_abs_phase_sensitivity'],label=name)
    axs[0,0].set_title('Central window photon intensity')
    axs[0,1].set_title('Signed central phase difference')
    axs[1,0].set_title('D / (I₀ + Iπ), central window')
    axs[1,1].set_title('Sum |D pixel| / sum (I₀ + Iπ)')
    for ax in axs.flat:
        ax.set_xlabel('time (fs)')
        ax.legend(fontsize=8)
    fig.savefig(out/'center_interference_diagnostics.png',dpi=170)
    plt.close(fig)

    if (out/'different_minus_same.npz').exists():
        dd=np.load(out/'different_minus_same.npz')
        fig,axs=plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
        axs[0].plot(dd['times_fs'],dd['delta_center_D'])
        axs[0].set_title('Difference between material conditions in central phase contrast')
        axs[0].set_ylabel('ΔD (probability)')
        axs[1].plot(dd['times_fs'],dd['delta_signed_contrast'])
        axs[1].set_ylabel('Difference in normalized contrast')
        axs[1].set_xlabel('time (fs)')
        fig.savefig(out/'spectral_only_difference.png',dpi=170)
        plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    w1=p.freq1/CM1;w2=p.freq2/CM1
    axs[0].plot(w1,p.J1/CM1,label='Material 1')
    axs[0].plot(w2,p.J2/CM1,label='Material 2')
    axs[0].set_xlabel('ω (cm⁻¹)')
    axs[0].set_ylabel('J(ω) (cm⁻¹)')
    axs[0].set_title('Discretized input spectral densities')
    axs[0].legend()
    axs[1].plot(w1,p.coupl1,label=f'Material 1: λ≈{p.reorg1/CM1:.1f} cm⁻¹')
    axs[1].plot(w2,p.coupl2,label=f'Material 2: λ≈{p.reorg2/CM1:.1f} cm⁻¹')
    axs[1].set_xlabel('ω (cm⁻¹)')
    axs[1].set_ylabel('c (mass-weighted atomic units)')
    axs[1].set_title('Couplings derived from J(ω)')
    axs[1].legend(fontsize=8)
    fig.savefig(out/'fitted_spectra_discretized.png',dpi=170)
    plt.close(fig)
    print('Plots written to',out.resolve())


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='input.txt')
    parser.add_argument('--results',default='results')
    a=parser.parse_args()
    visualize(a.results,a.input)
