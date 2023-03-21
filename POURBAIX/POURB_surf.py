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
       df2 = df.filter(['Time','TotEng'], axis=1)
    ax.plot(mean_df['Time'], mean_df['TotEng'], color='k', label = "Mean")
    ax.legend()
    if save_fig:
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    return mean_df

def _count_H_and_O(atoms, n_h2o=400, n_iro2=192): # Counts extra oxygens and hydrogens on surface
        ans            = atoms.get_atomic_numbers()
        unique, counts = np.unique(ans, return_counts=True)
        dict_z_count   = dict(zip(unique,counts))
        n_ir = dict_z_count[77] - n_iro2
        n_o  = dict_z_count[8]  - 2*n_iro2 - n_h2o
        try:
            n_h  = dict_z_count[1]  - 2*n_h2o
        except KeyError:
            print("No hydrogens found")
            n_h = 0
        return {"Ir": n_ir, "O": n_o, "H": n_h}

def prepare(init_strucs, log_mean_dfs, n_h2o=400, avg_last_n_frames=2000): #last 1000 = last 100 ps
    G_h2o  = -13.760  #-13.907 #eV
    G_h2   = -6.621 #-6.745  #eV
    dfs =    [df[-avg_last_n_frames:] for df in log_mean_dfs]
    std_dev = [round(df['TotEng'].std(), 2) for df in dfs]
    print("max-min:", [df['TotEng'].max()-df['TotEng'].min() for df in dfs])
    mixes =  [read(struc) for struc in init_strucs]
    stoich = [_count_H_and_O(mix, n_h2o=n_h2o) for mix in mixes]
    mean_surf_energy = [df['TotEng'].mean() for df in dfs]
    print(stoich)
    h2o_contri =    [sto['O']*G_h2o for sto in stoich]
    h2_contri  =    [(2*sto['O']-sto['H'])*0.5*G_h2 for sto in stoich]
    slopes =        [(2*sto['O']-sto['H'])*-1.0 for sto in stoich]
    intercept =     np.subtract(mean_surf_energy, h2o_contri)
    print("mean_surf_energy", mean_surf_energy, "\nh2o_contri", h2o_contri, "\nh2_conti", h2_contri)
    intercept =     np.add(intercept, h2_contri) 
    # divide by surface area
    # surf_areas = [mix.get_cell()[0][0]*mix.get_cell()[1][1] for mix in mixes]
    # intercept = np.divide(intercept, surf_areas)
    # intercept *= 1000
    # subtract by minimum intercept
    intercept -= min(intercept)
    return intercept, slopes, std_dev

def plot_pourbaix(intercept, slopes, std_dev, labels, xlim=(0,2), ylim=(-3,3), save_fig=False, fig_file='abc.png', fig_title=""):
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.set_xlim(xlim)
    axes.set_xticks(np.arange(xlim[0], xlim[1], 0.5), minor=True)
    axes.set_ylim(ylim)
    axes.set_yticks(np.arange(ylim[0], ylim[1], 0.5), minor=True)
    x_vals = np.array(axes.get_xlim())
    for i, (c, m, sig) in enumerate(zip(intercept, slopes, std_dev)):
        y_vals = c + m * x_vals
        axes.plot(x_vals, y_vals, label=labels[i])
        axes.fill_between(x_vals, y_vals-sig, y_vals+sig, alpha=0.3)
    axes.set(xlabel='$V_{SHE}$ $(V)$', ylabel='$Internal$ $energy$  $(eV)$')
    axes.grid(True, which='both', alpha=0.5)
    fig.suptitle(f'{fig_title}', fontsize=16)
    leg = axes.legend(loc='lower left', prop={'size': 15})#,  bbox_to_anchor=(1, 0.5), ncols=1)
    leg._legend_box.align = "left"
    if save_fig:
        plt.savefig(f'{fig_file}', dpi=300,  bbox_inches='tight')

# %%
def main():
    cwd = Path(os.getcwd())/"1_OH_vs_empty/WATER"
    init_strucs = glob.glob(str(cwd/"mix_*_0.xyz"))[:]
    init_strucs.sort()
    print(init_strucs)
    log_all_paths_prefix = [str(cwd/"logCut_OH_"), str(cwd/"logCut_emp_"),
                            str(cwd/"logCut_OOH_"), str(cwd/"logCut_O_")]
    log_all_paths_prefix.sort()
    log_mean_dfs = []
    for prefix in log_all_paths_prefix:
        log_mean_dfs.append(avg_logCuts(glob.glob(prefix+"*.lammps")[:8], 
                                        save_fig=False, 
                                        save_fig_path=f"./{prefix}_tot_eng.png"))

    c_, m_, sig_ = prepare(init_strucs=init_strucs, log_mean_dfs=log_mean_dfs, n_h2o=400)
    print("intercepts = ", c_, "\nslopes = ", m_, "\nStd. dev.", sig_)
    plot_pourbaix(intercept=c_, slopes=m_, std_dev=sig_, labels=['OH', 'OOH', 'O', 'clean'], 
                  save_fig=True, fig_file='O_OH_OOH_fill.png', fig_title='Water interface')

if __name__ == '__main__':
    main()
# %%
