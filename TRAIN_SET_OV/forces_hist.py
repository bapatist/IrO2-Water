#%%
import numpy as np
from ase.io import read
from matplotlib import pyplot as plt
import matplotlib
font = {'size'   : 18}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
#%%
take_absolute = True
train_set = read("./TRAIN_SET/low_F_strucs.xyz", ":")
# train_set = read("./TRAIN_SET/training_22.xyz", ":")
print(len(train_set), "No of strucs")
samples = [im for im in train_set if 'surf_sample' in im.info['config_type']]
ir_forces, o_forces, h_forces = [], [], []
for im in samples:
    for atom in im:
        id_atom = atom.index
        if atom.symbol == 'Ir':
            ir_forces.append(im.arrays['DFT_forces'][id_atom])
        if atom.symbol == 'O':
            o_forces.append(im.arrays['DFT_forces'][id_atom])
        if atom.symbol == 'H':
            h_forces.append(im.arrays['DFT_forces'][id_atom])
ir_forces, o_forces, h_forces = (np.hstack(ir_forces)), (np.hstack(o_forces)), (np.hstack(h_forces))
if take_absolute:
    ir_forces, o_forces, h_forces = np.abs(ir_forces), np.abs(o_forces), np.abs(h_forces)
# print("total force components of Iridum found: ", len(ir_forces))
print("Avg. of force components on Iridiums =", np.mean(ir_forces), 
      "\nAvg. of force components on Oxygens  =", np.mean(o_forces), 
      "\nAvg. of force components on Hydrogens=", np.mean(h_forces))


print("Std. of force components on Iridiums =", np.std(ir_forces), 
      "\nStd. of force components on Oxygens  =", np.std(o_forces), 
      "\nStd. of force components on Hydrogens=", np.std(h_forces))

#%%
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

def get_forces(atom_objects, take_absolute=True):
    forces_ir_dims, forces_ir_slab, forces_ir_inter = [], [], []
    forces_o_dims, forces_o_water, forces_o_slab, forces_o_inter = [], [], [], []
    forces_h_dims, forces_h_water, forces_h_slab, forces_h_inter = [], [], [], []
    def return_by_element(im,take_absolute):
        f_ir, f_o, f_h = [], [], []
        for atom in im:
                id_atom = atom.index
                if atom.symbol == 'Ir':
                    f_ir.append(im.arrays['DFT_forces'][id_atom])
                if atom.symbol == 'O':
                    f_o.append(im.arrays['DFT_forces'][id_atom])
                if atom.symbol == 'H':
                    f_h.append(im.arrays['DFT_forces'][id_atom])
        if f_ir:
            f_ir = np.hstack(f_ir)
        if f_o:
            f_o = np.hstack(f_o)
        if f_h:
            f_h = np.hstack(f_h)
        if take_absolute:
            f_ir, f_o, f_h = np.abs(f_ir), np.abs(f_o), np.abs(f_h)
        return f_ir, f_o, f_h
        
    for atom_obj in atom_objects:
        f_ir, f_o, f_h = return_by_element(atom_obj, take_absolute=take_absolute)
        if atom_obj.info['config_type'] == 'dimer':
            forces_ir_dims.append(f_ir)
            forces_o_dims.append(f_o)
            forces_h_dims.append(f_h)
        if atom_obj.info['config_type'] in ['water_MD_sample' , 'water_non_minima', 'water_minima']:
            forces_o_water.append(f_o)
            forces_h_water.append(f_h)              
        if atom_obj.info['config_type'][:4] == 'slab':
            forces_ir_slab.append(f_ir)
            forces_o_slab.append(f_o)
            if 'H' in atom_obj.get_chemical_formula():
                forces_h_slab.append(f_h)
        if atom_obj.info['config_type'][:4] == 'surf':
            forces_ir_inter.append(f_ir)
            forces_o_inter.append(f_o)
            forces_h_inter.append(f_h)
    try:
        forces_ir_dims, forces_ir_slab, forces_ir_inter = np.hstack(forces_ir_dims), np.hstack(forces_ir_slab), np.hstack(forces_ir_inter)
        forces_o_dims, forces_o_water, forces_o_slab, forces_o_inter = np.hstack(forces_o_dims), np.hstack(forces_o_water), np.hstack(forces_o_slab), np.hstack(forces_o_inter)
        forces_h_dims, forces_h_water, forces_h_slab, forces_h_inter = np.hstack(forces_h_dims), np.hstack(forces_h_water), np.hstack(forces_h_slab), np.hstack(forces_h_inter)
    except ValueError:
        print("WARNING: we got a value error while hstacking and we are skipping")            
    forces_all = [forces_ir_dims, forces_ir_slab, forces_ir_inter,
                  forces_o_dims, forces_o_water, forces_o_slab, forces_o_inter,
                  forces_h_dims, forces_h_water, forces_h_slab, forces_h_inter]
    labels = ['$\mathrm {Ir_{dims}}$', '$\mathrm {Ir_{slab}}$','$\mathrm {Ir_{inter}}$',
              '$\mathrm {O_{dims}}$', '$\mathrm {O_{water}}$','$\mathrm {O_{slab}}$','$\mathrm {O_{inter}}$',
              '$\mathrm {H_{dims}}$', '$\mathrm {H_{water}}$','$\mathrm {H_{slab}}$','$\mathrm {H_{inter}}$']
    return forces_all, labels

def plot_hist(forces, label, plot_cutoff=None):
    # print(np.shape(forces))
    forces = [f for f in forces]
    # colors = []
    # Plot the histogram
    if plot_cutoff:
        plt.xlim(-plot_cutoff, plot_cutoff)
    plt.hist(forces, stacked=True, bins='auto', 
             label=label)
    plt.xlabel('$\mathrm {F_{(x,\ y,\ z)\ DFT},\ eV/\AA}$')
    plt.ylabel('$\mathrm {Frequency}$')
    plt.legend()
    plt.savefig("PLOTs/forces_lowF.png", dpi=300, bbox_inches='tight')
#%%
forces_all, labels = get_forces(atom_objects=train_set, take_absolute=False)
forces_all = [forces_all[2]]+[forces_all[6]]+[forces_all[10]]
labels = [labels[2]]+[labels[6]]+[labels[10]]

plot_hist(forces=forces_all, label=labels, plot_cutoff=None)
# %%
