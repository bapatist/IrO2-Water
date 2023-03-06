# %%
import numpy as np
import pandas as pd
import glob, os
from matplotlib import pyplot as plt
from pathlib import Path
from ase.io import read
import matplotlib
font = {'size'   : 18}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
# %%
def _count_H_and_O(atoms, n_iro2=192, n_h2o=400): # Counts extra oxygens and hydrogens on surface
        ans            = atoms.get_atomic_numbers()
        unique, counts = np.unique(ans, return_counts=True)
        dict_z_count   = dict(zip(unique,counts))
        n_ir = dict_z_count[77] - n_iro2
        n_o  = dict_z_count[8]  - 2*n_iro2 - n_h2o
        n_h  = dict_z_count[1]  - 2*n_h2o
        return {"Ir": n_ir, "O": n_o, "H": n_h}

def prepare(init_strucs, log_files, avg_last_n_frames=1000): #last 1000 = last 100 ps
    G_h2o  = -13.907 #eV
    dfs =    [pd.read_table(file, delim_whitespace=True)[-avg_last_n_frames:] for file in log_files]
    mixes =  [read(struc) for struc in init_strucs]
    stoich = [_count_H_and_O(mix) for mix in mixes]
    mean_surf_energy = [df['TotEng'].mean() for df in dfs]

    h2o_contri =    [sto['O']*G_h2o for sto in stoich]
    slopes =        [(2*sto['O']-sto['H'])*-0.5 for sto in stoich]
    intercept =     np.subtract(mean_surf_energy, h2o_contri)
    return intercept, slopes

def plot_pourbaix(intercept, slopes, labels, xlim=(0, 20), save_fig=False):
    axes = plt.gca()
    axes.set_xlim(xlim)
    x_vals = np.array(axes.get_xlim())
    for i, (c, m) in enumerate(zip(intercept, slopes)):
        y_vals = c + m * x_vals
        axes.plot(x_vals, y_vals, label=labels[i].split('/')[-1][:-8].upper())
    axes.set(xlabel='$U_{RHE} (V)$', ylabel='$Internal$ $energy$  $(eV)$')
    leg = axes.legend(loc='upper right', prop={'size': 15})#,  bbox_to_anchor=(1, 0.5), ncols=1)
    leg._legend_box.align = "left"
    if save_fig:
        plt.savefig(f'./pourbaix.png', dpi=300,  bbox_inches='tight')
# %%
def main():
    n_surfaces = 2
    cwd = Path(os.getcwd())
    init_strucs = glob.glob(str(cwd/"*mix.xyz"))
    log_files    = glob.glob(str(cwd/"*logCut.lammps"))
    init_strucs.sort()
    log_files.sort()
    c_, m_ = prepare(init_strucs=init_strucs, log_files=log_files)
    print(c_, m_)
    plot_pourbaix(intercept=c_, slopes=m_, labels=init_strucs, save_fig=True)

if __name__ == '__main__':
    main()
# %%
