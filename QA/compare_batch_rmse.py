# %%
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
font = {'size'   : 18}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
#%%
def plot_bar(rmses, labels, title, ylabel):
    x = np.arange(len(labels))
    patterns = [''] * len(labels)
    patterns[3] = '//'
    plt.bar(x, rmses, color='cornflowerblue', edgecolor='black', width=0.5, hatch=patterns)
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.savefig(title, dpi=300, bbox_inches='tight')


labels = ['Interface', 'Water', 'Surface', 'All', 'Dimers']
#%% ENERGIES
# rmses_E  = [3.98, 3.99, 6.67, 22.2, 31.42]
rmses_E  = [2.42, 3.23, 4.70, 22.2, 41.0] #GAP_DFS
perc_E   = [0.06,0.06,  0.09, 0.61, 0.99]
# ylabel = '$\mathrm {RMSE\ energy, meV\ (per\ atom)}$'
ylabel = '$\mathrm {\%Error\ (Energy\ per\ atom)}$'
title  = "DFS_E_rmse.png"
plot_bar(rmses_E, labels, title, ylabel)
# %% FORCES
rmses_F = [174, 112, 746, 335, 632]
ylabel = '$\mathrm {RMSE\ forces,\ meV/\AA}$'
# ylabel = '$\mathrm {\%Error\ (Forces)}$'
title  = "compare_F_rmse.png"
plot_bar(rmses_F, labels, title, ylabel)
# %%
