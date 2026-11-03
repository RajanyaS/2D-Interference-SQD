"""Animation of phase-cycled interference for both material configurations.

python animate_interference.py --results results --output results/interference_evolution.gif
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def make_movie(path,outfile,fps=5):
    path=Path(path)
    sa=np.load(path/'same_interference.npz')
    di=np.load(path/'different_interference.npz')
    a=sa['Dmap'];b=di['Dmap'];d=b-a
    times=sa['times_fs']
    if len(a)!=len(b):raise ValueError('Phase responses must have same time grid')
    ext=(-b.shape[-1]/2,b.shape[-1]/2,b.shape[-2]/2,-b.shape[-2]/2)
    v=max(np.max(np.abs(a)),np.max(np.abs(b)),1e-10)
    vd=max(np.max(np.abs(d)),1e-11)
    fig,axs=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
    images=[]
    for i,(ar,title,vmax) in enumerate(((a,'Same materials: D',v),(b,'Different J: D',v),(d,'ΔD = different − same',vd))):
        im=axs[i].imshow(ar[0],origin='upper',extent=ext,cmap='RdBu_r',vmin=-vmax,vmax=vmax,interpolation='nearest',aspect='auto')
        axs[i].axhline(0,ls=':',color='grey',lw=.7)
        axs[i].set(title=title,xlabel='y index',ylabel='x index')
        fig.colorbar(im,ax=axs[i],shrink=.72)
        images.append(im)
    title=fig.suptitle(f'2D photon phase contrast (one excitation); t={times[0]:.2f} fs')
    def draw(i):
        for im,z in zip(images,(a,b,d)):im.set_data(z[i])
        title.set_text(f'2D photon phase contrast (one excitation); t={times[i]:.2f} fs')
        return (*images,title)
    movie=FuncAnimation(fig,draw,frames=len(times),interval=200,blit=False)
    outfile=Path(outfile)
    outfile.parent.mkdir(parents=True,exist_ok=True)
    movie.save(outfile,writer=PillowWriter(fps=fps))
    plt.close(fig)
    print('Saved',outfile)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',default='results')
    ap.add_argument('--output',default='results/interference_evolution.gif')
    ap.add_argument('--fps',type=int,default=5)
    args=ap.parse_args()
    make_movie(args.results,args.output,args.fps)
