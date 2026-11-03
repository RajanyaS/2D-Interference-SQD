"""Visualize an existing 2D SQD two-spectra ensemble; no new propagation required.

Run from project root:
 python preview_patterns.py --results results_distinct_preview_12traj
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def main(root):
    root=Path(root)
    s=np.load(root/'same_interference.npz')
    d=np.load(root/'different_interference.npz')
    times=s['times_fs']
    chosen=[24.2,43.5]
    inds=[int(np.argmin(abs(times-t))) for t in chosen]
    pos=(np.arange(s['Dmap'].shape[1])-s['Dmap'].shape[1]//2)
    ys=(np.arange(s['Dmap'].shape[2])-s['Dmap'].shape[2]//2)
    extent=(ys[0]-.5,ys[-1]+.5,pos[-1]+.5,pos[0]-.5)
    def heat(ax,data,vmax,header,subtitle,shared=False):
        im=ax.imshow(data,origin='upper',extent=extent,cmap='RdBu_r',norm=TwoSlopeNorm(vcenter=0,vmin=-max(vmax,1e-15),vmax=max(vmax,1e-15)),aspect='auto',interpolation='nearest')
        ax.axhline(-.5,ls='--',lw=1,color='black',alpha=.5)
        ax.set_title(header+'\n'+subtitle,fontsize=10)
        ax.set_xlabel('y (sites relative to center)')
        ax.set_ylabel('x (sites relative to center)')
        cb=plt.colorbar(im,ax=ax,shrink=.77,pad=.015);cb.ax.tick_params(labelsize=8)
    # Photon maps, 2 chosen times x 3 cases.
    fig,axes=plt.subplots(2,3,figsize=(15,10),constrained_layout=True)
    for r,ind in enumerate(inds):
        a=s['Dmap'][ind];b=d['Dmap'][ind];delta=b-a
        common=max(np.max(abs(a)),np.max(abs(b)))
        for j,(data,title,vmax) in enumerate([(a,'J₁ / J₁  :  D photon',common),(b,'J₁ / J₂  :  D photon',common),(delta,'Difference  :  ΔD photon',np.max(abs(delta))) ]):
            val=(float(np.sum(abs(a)) if j==0 else np.sum(abs(b)) if j==1 else np.sum(abs(delta))))
            heat(axes[r,j],data,vmax,title,f't={times[ind]:.1f} fs | total |map|={val:.2e}')
    fig.suptitle('Calculated 2D photon phase-contrast maps  D = I(0) − I(π)  |  12 thermal trajectories per case/phase',fontsize=15)
    fig.savefig(root/'preview_photon_interference_12traj.png',dpi=165)
    plt.close(fig)
    # Excitonic contrast from stored raw densities.
    sx0=np.load(root/'same_phi_0.npz');sxpi=np.load(root/'same_phi_pi.npz')
    dx0=np.load(root/'different_phi_0.npz');dxpi=np.load(root/'different_phi_pi.npz')
    Dsx=sx0['rhoX']-sxpi['rhoX'];Ddx=dx0['rhoX']-dxpi['rhoX']
    fig,axes=plt.subplots(2,3,figsize=(15,10),constrained_layout=True)
    for r,ind in enumerate(inds):
        a=Dsx[ind];b=Ddx[ind];delta=b-a
        common=max(np.max(abs(a)),np.max(abs(b)))
        for j,(data,title,vmax) in enumerate([(a,'J₁ / J₁  :  D exciton',common),(b,'J₁ / J₂  :  D exciton',common),(delta,'Difference  :  ΔD exciton',np.max(abs(delta))) ]):
            heat(axes[r,j],data,vmax,title,f't={times[ind]:.1f} fs | total |map|={np.sum(abs(data)):.2e}')
    fig.suptitle('Calculated 2D exciton phase-contrast maps | same 12-trajectory ensemble',fontsize=15)
    fig.savefig(root/'preview_exciton_interference_12traj.png',dpi=165)
    plt.close(fig)
    # Time traces and physical brightness guard.
    Sa=s['normalized_abs_phase_sensitivity'];Sb=d['normalized_abs_phase_sensitivity']
    Ba=s['center_brightness'];Bb=d['center_brightness'];
    fig,axs=plt.subplots(2,2,figsize=(12,8.5),constrained_layout=True)
    axs[0,0].plot(times,Sa,label='J₁/J₁',lw=2)
    axs[0,0].plot(times,Sb,label='J₁/J₂',lw=2)
    axs[0,0].set_ylabel('Normalized spatial phase sensitivity S')
    axs[0,0].set_title('Central-window normalized contrast')
    axs[0,1].plot(times,Ba,label='J₁/J₁',lw=2)
    axs[0,1].plot(times,Bb,label='J₁/J₂',lw=2)
    axs[0,1].set_ylabel('Central photon brightness B (probability)')
    axs[0,1].set_title('Actual photon signal in central region')
    axs[1,0].plot(times,Sb-Sa,color='darkmagenta',lw=2)
    axs[1,0].axhline(0,color='black',lw=.7)
    axs[1,0].set_ylabel('ΔS = S(J₁/J₂) − S(J₁/J₁)')
    axs[1,0].set_title('Material-induced change in S (preliminary)')
    axs[1,1].plot(times,d['center_D']-s['center_D'],color='darkorange',lw=2)
    axs[1,1].axhline(0,color='black',lw=.7)
    axs[1,1].set_ylabel('Δ central signed phase contrast')
    axs[1,1].set_title('Signed interference difference')
    for ax in axs.flat:
        for i in inds: ax.axvline(times[i],color='gray',ls=':',alpha=.55)
        ax.set_xlabel('Time (fs)')
        ax.grid(alpha=.18)
    axs[0,0].legend(frameon=False);axs[0,1].legend(frameon=False)
    fig.suptitle('12-trajectory numerical demonstration — not converged statistics',fontsize=14)
    fig.savefig(root/'preview_interference_time_traces_12traj.png',dpi=175)
    plt.close(fig)
    # Raw photon and exciton phase-conditioned maps at 24.2fs.
    ii=inds[0]
    fig,axs=plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
    x0= sx0['rhoX'][ii];xp=sxpi['rhoX'][ii];y0=dx0['rhoX'][ii];yp=dxpi['rhoX'][ii]
    cmax=max(np.max(z) for z in (x0,xp,y0,yp))
    for row,(label,d0,dp,component) in enumerate([('Exciton',sx0,sxpi,'rhoX'),('Photon',sx0,sxpi,'rhoC')]):
        same0=d0[component][ii];samepi=dp[component][ii];diff0=dx0[component][ii];diffpi=dxpi[component][ii]; v=max(np.max(z) for z in [same0,samepi,diff0,diffpi]);
        for col,(grid,title) in enumerate([(same0,'J₁/J₁, φ=0'),(samepi,'J₁/J₁, φ=π'),(diff0,'J₁/J₂, φ=0'),(diffpi,'J₁/J₂, φ=π')]):
            ax=axs[row,col]
            im=ax.imshow(grid,extent=extent,origin='upper',aspect='auto',cmap='magma',vmin=0,vmax=v,interpolation='nearest')
            ax.axhline(-.5,ls='--',lw=1,color='white',alpha=.6)
            ax.set_title(f'{label}: {title}',fontsize=10);ax.set_xlabel('y site');
            if col==0: ax.set_ylabel('x site')
            plt.colorbar(im,ax=ax,shrink=.75,pad=.015)
    fig.suptitle(f'Raw ensemble probability maps at {times[ii]:.2f} fs — 12 trajectories',fontsize=14)
    fig.savefig(root/'preview_raw_density_maps_12traj.png',dpi=155)
    plt.close(fig)
    # concise measurements
    idxA=inds[0];idxB=inds[1]
    for i in inds:
        print(f'At t={times[i]:.2f} fs: photon central brightness (same/diff) = {Ba[i]:.6g}/{Bb[i]:.6g};  S same/diff={Sa[i]:.5f}/{Sb[i]:.5f}; ΔS={Sb[i]-Sa[i]:+.5f};  max photon |ΔD|={np.max(np.abs(d["Dmap"][i]-s["Dmap"][i])):.3e}; max exciton |ΔD|={np.max(np.abs(Ddx[i]-Dsx[i])):.3e}')
    print('Generated 4 preview figure files in',root)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results',default='results_distinct_preview_12traj')
    args=parser.parse_args();main(args.results)
