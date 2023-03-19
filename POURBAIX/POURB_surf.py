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
def avg_logCuts(logFiles, save_fig=False, save_fig_path='tot_eng.png'):
    fig = plt.figure()
    dfs = [pd.read_table(file, delim_whitespace=True) for file in logFiles]
    dfs = [df.filter(['Time', 'TotEng']) for df in dfs]
    mean_df = sum(dfs)/len(dfs)
    ax = fig.add_subplot(111, xlabel="Time in ps", ylabel="Total Energy in eV")
    for df in dfs:
        ax.plot(df['Time'], df['TotEng'], alpha=0.2)
        # df2 = df.filter(['Time','TotEng'], axis=1)
    ax.plot(mean_df['Time'], mean_df['TotEng'], color='k', label = "Mean")
    ax.legend()
    if save_fig:
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    return mean_df

def _count_H_and_O(atoms, n_iro2=192, n_h2o=400): # Counts extra oxygens and hydrogens on surface
        ans            = atoms.get_atomic_numbers()
        unique, counts = np.unique(ans, return_counts=True)
        dict_z_count   = dict(zip(unique,counts))
        n_ir = dict_z_count[77] - n_iro2
        n_o  = dict_z_count[8]  - 2*n_iro2 - n_h2o
        n_h  = dict_z_count[1]  - 2*n_h2o
        return {"Ir": n_ir, "O": n_o, "H": n_h}

def prepare(init_strucs, log_mean_dfs, avg_last_n_frames=1000): #last 1000 = last 100 ps
    G_h2o  = -13.907 #eV
    G_h2   = -6.745  #eV
    dfs =    [df[-avg_last_n_frames:] for df in log_mean_dfs]
    mixes =  [read(struc) for struc in init_strucs]
    stoich = [_count_H_and_O(mix) for mix in mixes]
    mean_surf_energy = [df['TotEng'].mean() for df in dfs]
    print(stoich)
    h2o_contri =    [sto['O']*G_h2o for sto in stoich]
    h2_contri  =    [(2*sto['O']-sto['H'])*0.5*G_h2 for sto in stoich]
    slopes =        [(2*sto['O']-sto['H'])*-0.5 for sto in stoich]
    intercept =     np.subtract(mean_surf_energy, h2o_contri)
    intercept =     np.add(intercept, h2_contri) 
    # subtract by minimum intercept
    intercept -= min(intercept)
    # divide by surface area
    # surf_areas = [mix.get_cell()[0][0]*mix.get_cell()[1][1] for mix in mixes]
    # intercept = np.divide(intercept, surf_areas)
    # intercept *= 1000
    return intercept, slopes

def plot_pourbaix(intercept, slopes, labels, xlim=(0, 2), save_fig=False):
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.set_xlim(xlim)
    x_vals = np.array(axes.get_xlim())
    for i, (c, m) in enumerate(zip(intercept, slopes)):
        y_vals = c + m * x_vals
        axes.plot(x_vals, y_vals, label=labels[i].split('/')[-1][:-8].upper())
    axes.set(xlabel='$V_{SHE}$ $(V)$', ylabel='$Internal$ $energy$  $(eV)$')
    leg = axes.legend(loc='upper right', prop={'size': 15})#,  bbox_to_anchor=(1, 0.5), ncols=1)
    leg._legend_box.align = "left"
    if save_fig:
        plt.savefig(f'./pourbaix.png', dpi=300,  bbox_inches='tight')

# %%
def main():
    cwd = Path(os.getcwd())
    init_strucs = glob.glob(str(cwd/"1_OH_vs_empty/mix_*_0.xyz"))
    init_strucs.sort()
    log_all_paths_prefix = ["1_OH_vs_empty/logCut_OH_", "1_OH_vs_empty/logCut_emp_"]
    log_all_paths_prefix.sort()
    log_mean_dfs = []
    for prefix in log_all_paths_prefix:
        log_mean_dfs.append(avg_logCuts(glob.glob(prefix+"*.lammps"), save_fig=False, save_fig_path=f"./{prefix}_tot_eng.png"))

    c_, m_ = prepare(init_strucs=init_strucs, log_mean_dfs=log_mean_dfs)
    print(c_, m_)
    plot_pourbaix(intercept=c_, slopes=m_, labels=init_strucs, save_fig=True)

if __name__ == '__main__':
    main()
#%%
