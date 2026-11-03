"""SQD-style sequential runner for 2D MFE two-arm interference.

Run 'same' and 'different' with identical incident state, nuclear seeds, phase
conditions, detection window and all parameters except the specified baths.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from Model.polaritonmod import parameters, CM1, AU_TO_FS
from initialMC import initElectronic, initial_state
from Method.split_op_Saeed import runTraj


def parse_phases(txt):
    values=[]
    for v in txt.split(','):
        v=v.strip().lower()
        if v in ('pi','π'):values.append(('pi',np.pi))
        elif v in ('pi/2','halfpi'):values.append(('pi2',np.pi/2))
        elif v in ('3pi/2','threepi2'):values.append(('3pi2',3*np.pi/2))
        elif v in ('0','zero'):values.append(('0',0.))
        else:values.append((v.replace('.','p'),float(v)))
    return values


def center_window(par):
    win=np.zeros((par.Nx,par.Ny),dtype=bool)
    midx=par.Nx//2
    midy=par.Ny//2
    win[max(0,midx-par.center_halfwidth_x):min(par.Nx,midx+par.center_halfwidth_x+1),
        max(0,midy-par.center_halfwidth_y):min(par.Ny,midy+par.center_halfwidth_y+1)]=True
    return win


def compute_phase_diagnostics(case_files,window,output):
    data={s:np.load(fname) for s,fname in case_files.items()}
    save={'times_fs':data[next(iter(data))]['times_au']*AU_TO_FS}
    for key,dat in data.items():
        rho=dat['rhoC']
        save[f'center_I_{key}']=np.sum(rho[:,window],axis=1)
    if '0' in data and 'pi' in data:
        i0,ipi=data['0']['rhoC'],data['pi']['rhoC']
        D=i0-ipi
        sumI=i0+ipi
        denom=np.sum(sumI[:,window],axis=1)
        dsum=np.sum(D[:,window],axis=1)
        dabs=np.sum(np.abs(D[:,window]),axis=1)
        # Numerical epsilon keeps empty initial window finite and indicates no signal.
        save.update(Dmap=D, center_D=dsum,center_abs_D=dabs,
            center_brightness=denom,
            normalized_signed_contrast=np.divide(dsum,denom,out=np.zeros_like(dsum),where=denom>1e-12),
            normalized_abs_phase_sensitivity=np.divide(dabs,denom,out=np.zeros_like(dabs),where=denom>1e-12))
    if all(k in data for k in ('0','pi2','pi','3pi2')):
        samples=np.stack([data[k]['rhoC'] for k in ('0','pi2','pi','3pi2')])
        a=(samples[0]-samples[2])/2
        b=(samples[1]-samples[3])/2
        mean=np.mean(samples,axis=0)
        # Operational first harmonic: I(phi) ~= Ibar + a*cos(phi)+b*sin(phi).
        # MFE need not give exactly sinusoidal phase dependence.
        mag=np.sqrt(a*a+b*b)
        Icentral=np.sum(samples[:,:,window],axis=2)
        save['four_phase_first_harmonic_map']=mag
        save['four_phase_mean_map']=mean
        save['four_phase_center_sampled_visibility']=(
            (np.max(Icentral,axis=0)-np.min(Icentral,axis=0)) /
            np.maximum(np.max(Icentral,axis=0)+np.min(Icentral,axis=0),1e-15))
    np.savez_compressed(output,**save)
    return save


def main(input_path='input.txt',outdir='results',traj_start=0,traj_count=None):
    par=parameters.from_file(input_path)
    # A cluster job can average just its own non-overlapping global seed range.
    if traj_start < 0: raise ValueError('traj_start must be >= 0')
    batch_count = par.NTraj if traj_count is None else int(traj_count)
    if batch_count < 1: raise ValueError('traj_count must be >= 1')
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    print('Grid',par.Nx,par.Ny,'; states',2*par.Nsite,'; phonon modes/site',par.n_modes)
    print('J-derived reorganization energies: material 1 %.2f cm-1, material 2 %.2f cm-1'
          %(par.reorg1/CM1,par.reorg2/CM1))
    print('Two wavepackets = one coherent excitation, not two excitations')
    XL,CL,XR,CR=initElectronic(par)
    ov=np.vdot(XL,XR)+np.vdot(CL,CR)
    np.savez_compressed(out/'initial_packets.npz',XL=XL,CL=CL,XR=XR,CR=CR,overlap=ov)
    print('Initial <L|R>:',ov)
    win=center_window(par)
    np.save(out/'center_mask.npy',win)
    summary={'settings':{'Nx':par.Nx,'Ny':par.Ny,'Nsite':par.Nsite,
                         'dtN_au':par.dtN,'NSteps':par.NSteps,'NTraj':par.NTraj,
                         'traj_start':traj_start,'traj_count':batch_count,
                         'input_sha256':hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                         'SEED':par.SEED,
                         'spectra':[par.spectrum1,par.spectrum2],
                         'lambda1_discretized_cm1':par.reorg1/CM1,
                         'lambda2_discretized_cm1':par.reorg2/CM1,
                         'input_overlap':{'real':float(ov.real),'imag':float(ov.imag)}}}
    saved={}
    for case in [v.strip() for v in par.cases.split(',') if v.strip()]:
        saved[case]={}
        for label,phi in parse_phases(par.phases):
            psi=initial_state(XL,CL,XR,CR,phi)
            avg=None
            for itraj in range(traj_start,traj_start+batch_count):
                # Identical seed for the same trajectory number across phases and cases.
                # Actual Q dynamics then differs because electronic forces differ.
                seed=par.SEED+itraj
                res=runTraj(par,case,psi,seed)
                if avg is None:
                    avg={key:val.copy() if hasattr(val,'copy') else val for key,val in res.items()}
                    for key in ('rhoX','rhoC','norm','energy'): avg[key]=res[key].copy()/batch_count
                else:
                    for key in ('rhoX','rhoC','norm','energy'):avg[key]+=res[key]/batch_count
            outfile=out/f'{case}_phi_{label}.npz'
            np.savez_compressed(outfile,**avg)
            saved[case][label]=str(outfile)
            norm_error=float(max(abs(avg['norm']-1)))
            energy_drift=float(max(abs(avg['energy']-avg['energy'][0])))
            summary[f'{case}_{label}']={'max_norm_error':norm_error,'max_energy_drift_Hartree':energy_drift,
                                       'total_photon_final':float(np.sum(avg['rhoC'][-1]))}
            print(case,label,'norm_err',f'{norm_error:.2e}','energy_drift_au',f'{energy_drift:.2e}')
            if norm_error > par.assert_norm_tol:raise ArithmeticError('Norm drift too high')
        compute_phase_diagnostics(saved[case],win,out/f'{case}_interference.npz')
    if 'same' in saved and 'different' in saved and all(x in saved['same'] and x in saved['different'] for x in ('0','pi')):
        a=np.load(out/'same_interference.npz')
        b=np.load(out/'different_interference.npz')
        np.savez_compressed(out/'different_minus_same.npz',
            times_fs=a['times_fs'],delta_Dmap=b['Dmap']-a['Dmap'],
            delta_center_D=b['center_D']-a['center_D'],
            delta_center_brightness=b['center_brightness']-a['center_brightness'],
            delta_signed_contrast=b['normalized_signed_contrast']-a['normalized_signed_contrast'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('Saved raw ensembles and interferometric diagnostics to',out.resolve())


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='input.txt')
    ap.add_argument('--output',default='results')
    ap.add_argument('--traj-start',type=int,default=0,
                    help='First global thermal trajectory index (used for seeds)')
    ap.add_argument('--traj-count',type=int,default=None,
                    help='Number of trajectories in this job; default input NTraj')
    a=ap.parse_args()
    main(a.input,a.output,a.traj_start,a.traj_count)
