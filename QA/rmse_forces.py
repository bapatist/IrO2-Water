
#%%
import numpy as np
from ase.io import read
from wfl.plotting.plot_ef_correlation import extract_energies_per_atom, plot_energy, plot, extract_forces, plot_force_components, rms_text_for_plots
from matplotlib import pyplot as plt
import matplotlib
font = {'size': 18}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
# %% FORCE ERROR AND SCATTER
def plot_scatter(x,y, gap_or_mace):
    print(f"Total points used for scatter = {len(x)}")
    fig, ax = plt.subplots()
    ax.scatter(x,y)
    # x=y line
    for_limits = np.concatenate((x,y))
    elim = (for_limits.min() - 0.01, for_limits.max() + 0.01)
    ax.set_xlim(elim)
    ax.set_ylim(elim)
    ax.plot(elim, elim, c='k')
    # set labels
    ax.set_ylabel(f'{gap_or_mace} forces in eV/\AA')
    ax.set_xlabel('DFT forces in eV/\AA')
    fig.set_size_inches(8, 6.5)

def _rms_ener(x_ref, x_pred, MAE=False):
    """ Takes two datasets of the same shape and returns a dictionary containing RMS error data"""
    x_ref  = np.array(x_ref)
    x_pred = np.array(x_pred)
    if np.shape(x_pred) != np.shape(x_ref):
        raise ValueError('WARNING: not matching shapes in rms')
    if MAE:
        print("WARNING: getting MAE not RMSE@")
        error_2 = (x_ref - x_pred)
        average = (np.mean(error_2))
    else: 
        error_2 = (x_ref - x_pred) ** 2
        average = np.sqrt(np.average(error_2))
    std_    = np.sqrt(np.var(error_2))
    return {'rmse': average, 'std': std_}

def _perc_err(ground_truth, predictions, low_cap = 1.0):
    absolute_error = np.abs(predictions-ground_truth)
    for i, val in enumerate(ground_truth):
        if val<low_cap:
            ground_truth[i] = low_cap
    percentage_error = (absolute_error / np.abs(ground_truth)) * 100
    return format(np.mean(percentage_error), '.2f')
#%%
def rmse_forces(samples_file, gap_or_mace="GAP", config_type=None, error_type="rmse", plot=True):
    train_set = read(samples_file, ":")
    if config_type!=None:
        train_set = [s for s in train_set if config_type in s.info['config_type']]
    config_types = [im.info['config_type'] for im in train_set]

    print(f"Set has total of {len(train_set)} number of structures")
    GAP_forces, DFT_forces = [], []
    for im in train_set:
        GAP_forces.append(np.hstack(im.arrays[f'{gap_or_mace}_forces']))
        DFT_forces.append(np.hstack(im.arrays['DFT_forces']))
    GAP_forces, DFT_forces = (np.hstack(GAP_forces)), (np.hstack(DFT_forces))
    if error_type=="rmse":
        print("For training set having", 
              f"{len(train_set)} structures \n", 
              f"containing config types: {set(config_types)}\n",
              f"RMSE forces= {_rms_ener(DFT_forces, GAP_forces)}")
    if error_type=="percentage":
        print("For training set having", 
              f"{len(train_set)} structures \n", 
              f"containing config types: {set(config_types)}\n",
              f"% error forces= {_perc_err(DFT_forces, GAP_forces)}")
    if plot:
        plot_scatter(DFT_forces, GAP_forces, gap_or_mace=gap_or_mace)
    # if save_plot:
    #     plt.savefig(save, dpi=300, bbox_inches='tight')

#%% FORCE RMSEs
# options for config type: 'surf_sample', 'water', 'slab', 'dimer', None=All
gap_or_mace = "GAP"
rmse_forces(gap_or_mace=gap_or_mace,
            samples_file="TEST_SET/I4/test_lowF.xyz",
            # samples_file="TRAIN_SET/I4/ts_ALL.xyz",
            config_type='surf',
            error_type="rmse",
            plot=True) # "percentage" or "rmse"
# %%
rmse_forces(gap_or_mace=gap_or_mace,
            samples_file="MACE_TRAIN+TEST/test_all_I4_MACE_GPU.xyz",
            config_type='surf',
            error_type="rmse",
            plot=True) # "percentage" or "rmse"
# %%
