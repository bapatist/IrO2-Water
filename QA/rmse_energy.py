#%%
import numpy as np
from ase.io import read
from wfl.plotting.plot_ef_correlation import extract_energies_per_atom, plot_energy, plot, extract_forces, plot_force_components, rms_text_for_plots
from matplotlib import pyplot as plt
import matplotlib
font = {'size': 18}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
# %%
def _perc_err(ground_truth, predictions):
    absolute_error = np.abs(predictions-ground_truth)
    percentage_error = (absolute_error / np.abs(ground_truth)) * 100
    return format(np.mean(percentage_error), '.2f')

def rms_ener(x_ref, x_pred):
    """ Takes two datasets of the same shape and returns a dictionary containing RMS error data"""
    x_ref  = np.array(x_ref)
    x_pred = np.array(x_pred)
    if np.shape(x_pred) != np.shape(x_ref):
        raise ValueError('WARNING: not matching shapes in rms')
    error_2 = (x_ref - x_pred) ** 2
    average = np.sqrt(np.average(error_2))
    std_    = np.sqrt(np.var(error_2))
    return {'rmse': average, 'std': std_}
#%%
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
    ax.set_ylabel(f'{gap_or_mace} energy in eV')
    ax.set_xlabel('DFT energy in eV')
    fig.set_size_inches(8, 6.5)
#%%
def get_rmse_energy(samples_file, config_type=None, test_set_file=None, gap_or_mace="GAP"):
    train_set = read(samples_file, ":")
    if config_type!=None:
        train_set = [s for s in train_set if config_type in s.info['config_type']]
    config_types = [im.info['config_type'] for im in train_set]

    gap_ener_train = extract_energies_per_atom(train_set, f"{gap_or_mace}_energy", e0 = None)
    dft_ener_train = extract_energies_per_atom(train_set, "DFT_energy", e0 = None)
    print("For training set having", 
          f"{len(train_set)} structures \n", 
          f"containing config types: {set(config_types)}\n",
          f"RMSE energy= {rms_ener(dft_ener_train, gap_ener_train)}")
    plot_scatter(gap_ener_train, dft_ener_train, gap_or_mace=gap_or_mace)
    if test_set_file:
        test_set = read(test_set_file, ":")
        gap_ener_test  = extract_energies_per_atom(test_set, f"{gap_or_mace}_energy", e0 = None)
        dft_ener_test  = extract_energies_per_atom(test_set, "DFT_energy", e0 = None)
        print("For test set having", f"{len(test_set)} structures \n",
            f"RMSE energy= {rms_ener(dft_ener_test, gap_ener_test)}")
        # plot_scatter(gap_ener_test, dft_ener_test)

def get_perc_err_energy(samples_file, config_type=None, test_set_file=None, gap_or_mace="GAP"):
    train_set = read(samples_file, ":")
    if config_type!=None:
        train_set = [s for s in train_set if config_type in s.info['config_type']]
    config_types = [im.info['config_type'] for im in train_set]

    gap_ener_train = np.array(extract_energies_per_atom(train_set, f"{gap_or_mace}energy", e0 = None))
    dft_ener_train = np.array(extract_energies_per_atom(train_set, "DFT_energy", e0 = None))
    # print(len(gap_ener_train), len(dft_ener_train))
    print("For training set having",
          f"{len(train_set)} structures \n", 
          f"containing config types: {set(config_types)}\n",
          f"% error= {_perc_err(dft_ener_train, gap_ener_train)}")
    # plot_scatter(gap_ener_train, dft_ener_train)
    if test_set_file:
        test_set = read(test_set_file, ":")
        gap_ener_test  = extract_energies_per_atom(test_set, f"{gap_or_mace}energy", e0 = None)
        dft_ener_test  = extract_energies_per_atom(test_set, "DFT_energy", e0 = None)
        print("For test set having", f"{len(test_set)} structures \n",
               f"% error energy= {_perc_err(dft_ener_test, gap_ener_test)}")
        plot_scatter(gap_ener_test, dft_ener_test, gap_or_mace=gap_or_mace)
# RMSEs ENERGY
#%%
# options for config type: 'surf_sample', 'water', 'slab', 'dimer'
file        = "MACE_TRAIN+TEST/test_all_I4_MACE_GPU.xyz"
# file        = "TEST_SET/I4/test_lowF.xyz"
config_type = 'surf'
gap_or_mace = "MACE"
get_rmse_energy(config_type=config_type,
                samples_file=file,
                gap_or_mace=gap_or_mace)
# get_perc_err_energy(config_type=config_type,
#                     train_set_file=ts_file)
# %%