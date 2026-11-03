"""Inspect input Brownian J1/J2 and discrete oscillator energies before running."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from Model.polaritonmod import parameters, CM1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input_brownian.txt')
    parser.add_argument('--out', default='results_brownian/brownian_fit_discretization.png')
    args = parser.parse_args()
    p = parameters.from_file(args.input)
    wcm = np.linspace(0.1, max(2600., p.omega_max_cm1+300.), 4000)
    wa = wcm*CM1
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9,7), sharex=True)
    for s,c,colors in [('1',p.coupl1,'C0'),('2',p.coupl2,'C1')]:
        ax.plot(wcm, p.brownian_spectrum(wa, s), color=colors,
                label=f'Material {s} exact fitted J(w)')
        f = p.freq1 if s=='1' else p.freq2
        ax2.stem(f/CM1, c*c/(2*f*f)/CM1, linefmt=colors+'-',
                 markerfmt=colors+'o',basefmt=' ',
                 label=f'Material {s} discrete reorganization weights (cm^-1)')
    ax.set_ylabel('J(omega) [Hartree, if fit units valid]')
    ax.legend()
    ax.grid(alpha=.25)
    ax2.set_xlabel('Frequency (cm$^{-1}$)')
    ax2.set_ylabel('Each mode lambda_j (cm$^{-1}$)')
    ax2.legend()
    ax2.grid(alpha=.25)
    fig.tight_layout()
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=160)
    plt.close(fig)
    print('Saved:',out)
    print('Reorganization (discrete): lambda_1 =',p.reorg1/CM1,'cm^-1; lambda_2 =',p.reorg2/CM1,'cm^-1')

if __name__=='__main__':
    main()
