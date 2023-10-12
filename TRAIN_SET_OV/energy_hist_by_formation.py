# %%
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
font = {'size'   : 18}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
from ase.io import read, write

# %%
def conf_categories(config_types):
    n_mono, n_dim, n_wat, n_vac, n_surf = 5*[0]
    for c_type in config_types:
        if c_type == 'isolated_atom':
            n_mono += 1
        elif c_type == 'dimer':
            n_dim +=1
        elif c_type in ['water_MD_sample' , 'water_non_minima', 'water_minima']:
            n_wat +=1
        elif c_type[:4] in ['slab', 'bulk']:
            n_vac +=1
        elif c_type[:4] in ['surf']:
            n_surf +=1        
    return (n_mono, n_dim, n_wat, n_vac, n_surf)

def _calc_formation_energy(im, monomer_Es):
    total_energy   = im.info['DFT_energy']
    unique, counts = np.unique(im.get_atomic_numbers(), 
                               return_counts=True)
    counts_dict    = dict(zip(unique, counts))
    ir_Es, o_Es, h_Es = 0, 0, 0
    if 77 in counts_dict.keys():
        ir_Es      = monomer_Es['Ir']*counts_dict[77]
    if 8 in counts_dict.keys():
        o_Es       = monomer_Es['O']*counts_dict[8]
    if 1 in counts_dict.keys():
        h_Es       = monomer_Es['H']*counts_dict[1]
    formation_E    = total_energy - ir_Es - o_Es - h_Es
    formation_E   /= im.get_global_number_of_atoms()
    return formation_E

def plot_energy_histogram(atom_objects, plot_positives=True, plot_negatives=True, plot_name=None):
    # List to store per-atom energies
    per_atom_energies_dims = []
    per_atom_energies_wat = []
    per_atom_energies_slab = []
    per_atom_energies_surf  = []
    monomer_Es = {}
    for im in atom_objects:
        if im.info['config_type']=='isolated_atom':
            if im.get_chemical_formula()=="Ir":
                monomer_Es['Ir'] = im.info['DFT_energy']
            elif im.get_chemical_formula()=="O":
                monomer_Es['O'] = im.info['DFT_energy']
            elif im.get_chemical_formula()=="H":
                monomer_Es['H'] = im.info['DFT_energy']
    print(monomer_Es)
    # Calculate per-atom energies
    for atom_obj in atom_objects:
        per_atom_formate_energy = _calc_formation_energy(im=atom_obj, monomer_Es=monomer_Es)
        if not plot_positives:
            if per_atom_formate_energy>0:
                continue
        if not plot_negatives:
            if per_atom_formate_energy<0:
                continue

        if atom_obj.info['config_type'] == 'dimer':
            per_atom_energies_dims.append(per_atom_formate_energy)
        if atom_obj.info['config_type'] in ['water_MD_sample' , 'water_non_minima', 'water_minima']:
            per_atom_energies_wat.append(per_atom_formate_energy)
        if atom_obj.info['config_type'][:4] == 'slab':
            per_atom_energies_slab.append(per_atom_formate_energy)
        if atom_obj.info['config_type'][:4] == 'surf':
            per_atom_energies_surf.append(per_atom_formate_energy)
    per_atom_energies = [per_atom_energies_dims, per_atom_energies_wat,
                         per_atom_energies_slab, per_atom_energies_surf]
    labels = ['$\mathrm {Dimers}$', 
              '$\mathrm {Water}$', '$\mathrm {Surface}$', '$\mathrm {Interface}$']
    # Plot the histogram
    # plt.xlim(0, 35)
    plt.hist(per_atom_energies, stacked=True, bins='auto', 
             label=labels, edgecolor=None)
    plt.xlabel('$\mathrm {E_{DFT},\ eV\ (per\ atom)}$')
    plt.ylabel('$\mathrm {Frequency}$')
    plt.legend()
    if plot_name:
        plt.savefig(plot_name, dpi=300, bbox_inches='tight')

# %%

ts = read("./TRAIN_SET/low_F_TRAINING_ITER4_20_WITHOUTGAP.xyz", ":")
# ts = read("./TRAIN_SET/training_22.xyz", ":")
plot_name = "PLOTs/Ehist_I4_lowF.png"
print(f"Training set as {len(ts)} total structures")

config_types = [im.info['config_type'] for im in ts]
print(set(config_types))
# conf_categories(config_types=config_types)
plot_energy_histogram(ts, plot_negatives=True, plot_name=plot_name)
# %%




# %%
n_HO_dims = len([im for im in ts if len(im)==2 and 77 not in im.get_atomic_numbers()])
n_Ir_dims = len([im for im in ts if len(im)==2 and 77 in im.get_atomic_numbers()])
n_wat_mins= len([im for im in ts if im.info['config_type']=="water_minima"])
n_wat_nonmins= len([im for im in ts if im.info['config_type']=="water_non_minima"])
n_wat_md= len([im for im in ts if im.info['config_type']=="water_MD_sample"])
n_slab_vac= len([im for im in ts if im.info['config_type']=="slab_covs"])

print("Dimers with H and O = ", n_HO_dims)
print("Dimers with Ir (Ir-O and Ir-Ir)= ", n_Ir_dims)
print("Water minimas = ", n_wat_mins)
print("Water non-minimas = ", n_wat_nonmins)
print("Water MD samples = ", n_wat_md)
print("Slab = ", n_slab_vac)
# %%
