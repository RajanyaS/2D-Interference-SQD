"""Visual proof of SQD localization of two LP wavepackets (no Gaussian target)."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from Model.polaritonmod import parameters
from initialMC import sqd_mc_coefficients, _setup, _distances2, _LP_fields


def make_plot(par, path):
    fig,ax=plt.subplots(2,3,figsize=(13,6.5),constrained_layout=True)
    for row,side in enumerate(('left','right')):
        Ek,u,v,allowed,x0,y0,*_= _setup(par,side)
        KX,KY=np.meshgrid(par.kx,par.ky,indexing='ij')
        init=np.where(allowed,np.exp(-1j*(KX*(x0-par.x[0])+KY*(y0-par.y[0]))),0).astype(complex)
        init/=np.linalg.norm(init)
        best,u,v,allowed,info=sqd_mc_coefficients(par,side,par.initial_seed+(0 if row==0 else 11))
        for column,aa in enumerate((init,best)):
            X,C=_LP_fields(aa,u,v)
            rho=abs(X)**2+abs(C)**2
            im=ax[row,column].imshow(rho.T,origin='lower',aspect='auto',
                  extent=(par.x[0]/par.a_bohr,par.x[-1]/par.a_bohr,
                          par.y[0]/par.a_bohr,par.y[-1]/par.a_bohr))
            ax[row,column].axvline(x0/par.a_bohr,c='red',ls='--',lw=1)
            ax[row,column].axvline(0,c='white',ls=':',lw=1)
            ax[row,column].set_title(f'{side}: {"Before MC" if column==0 else "After MC"}')
            ax[row,column].set_xlabel('x / lattice sites')
            ax[row,column].set_ylabel('y / lattice sites')
            fig.colorbar(im,ax=ax[row,column],shrink=.74,label='Probability / site')
        ax[row,2].imshow(allowed.T,origin='lower',aspect='auto',interpolation='nearest')
        ax[row,2].set_title(f'{side} allowed LP modes: {info["n_allowed"]}')
        ax[row,2].set_xlabel('FFT kx index')
        ax[row,2].set_ylabel('FFT ky index')
        ax[row,2].text(.02,.02,
            f'SQD spread: {info["initial_spread_bohr2"]/par.a_bohr**2:.2f} → '
            f'{info["final_spread_bohr2"]/par.a_bohr**2:.2f} sites²\n'
            f'accepted {info["accepted"]}/{info["ntry"]} moves',
            transform=ax[row,2].transAxes,va='bottom',ha='left',fontsize=9,
            bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
    fig.suptitle('2D SQD energy-shell localization: two inward-moving LP arms',fontsize=14)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=170)
    plt.close(fig)
    print(f'Saved {path}')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='input.txt')
    ap.add_argument('--output',default='results/sqd_mc_localization.png')
    args=ap.parse_args()
    make_plot(parameters.from_file(args.input),args.output)
