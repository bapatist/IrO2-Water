# %%
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
font = {'size'   : 18}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
from ase.io import read, write
#%%
ts = read("training_22.xyz", ":")
print(f"Training set as {len(ts)} total structures")
# %%
ts_high, ts_low = [], []
for im in ts:
    if im.info['config_type'] == 'isolated_atom':
        ts_low.append(im)
        continue
    energy = im.info['DFT_energy']
    num_atoms = im.get_global_number_of_atoms()
    per_atom_energy = energy / num_atoms
    if per_atom_energy > -3:
        ts_high.append(im)
    else:
        ts_low.append(im)
print(len(ts_high), "high E strucs\n",
      len(ts_low), "low E strucs\n")

#%%
def plot_energy_histogram(atom_objects, figname=None):
    # List to store per-atom energies
    per_atom_energies_dims = []
    per_atom_energies_wat = []
    per_atom_energies_slab = []
    per_atom_energies_surf  = []
    # Calculate per-atom energies
    for atom_obj in atom_objects:
        energy = atom_obj.info['DFT_energy']
        num_atoms = atom_obj.get_global_number_of_atoms()
        per_atom_energy = energy / num_atoms
        if atom_obj.info['config_type'] == 'dimer':
            per_atom_energies_dims.append(per_atom_energy)
        if atom_obj.info['config_type'] in ['water_MD_sample' , 'water_non_minima', 'water_minima']:
            per_atom_energies_wat.append(per_atom_energy)
        if atom_obj.info['config_type'][:4] == 'slab':
            per_atom_energies_slab.append(per_atom_energy)
        if atom_obj.info['config_type'][:4] == 'surf':
            per_atom_energies_surf.append(per_atom_energy)
    per_atom_energies = [per_atom_energies_dims, per_atom_energies_wat,
                         per_atom_energies_slab, per_atom_energies_surf]
    labels = ['$\mathrm {Dimers}$', 
              '$\mathrm {Water}$', '$\mathrm {Surface}$', '$\mathrm {Interface}$']
    # Plot the histogram
    # plt.xlim(0, 35)
    plt.hist(per_atom_energies, stacked=True, bins='auto', 
             label=labels, edgecolor='black')
    plt.xlabel('$\mathrm {E_{DFT},\ eV\ (per\ atom)}$')
    plt.ylabel('$\mathrm {Frequency}$')
    plt.legend()
    # if figname:
        # plt.savefig(figname, dpi=300, bbox_inches='tight')
#%%
# plot_energy_histogram(ts_high, figname="highE_hist.png")
plot_energy_histogram(ts_low, figname="lowE_hist.png")
# %%
write("ts_lowE.xyz", ts_low)
write("ts_highE.xyz", ts_high)
# %%
