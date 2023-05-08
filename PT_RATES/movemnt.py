# %%
from ase.io import read
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
font = {'size'   : 18}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
# %%
md = read("8oh_newmd.traj", ":")
init_struc = read("8oh_mix.xyz")
# %%
H_surf = [atom.index for atom in init_struc if atom.z==13.77091714]
print("Total surface H atoms of OH = ", len(H_surf))
print("Initial positions of surface OH, H atoms ", init_struc.get_positions(H_surf), "\nAtom Indices = ", H_surf)
positions = [im[H_surf].get_positions() for im in md]
positions = np.array(positions)
max_Ir_Z = max([atom.z for atom in init_struc if atom.symbol == 'Ir'])
#These Iridiums are expected to stay put
Ir_bri = np.array([atom.position for atom in init_struc if (atom.symbol == 'Ir') and
                   (atom.z == max_Ir_Z)])
Ir_cus = np.array([atom.position for atom in init_struc if (atom.symbol == 'Ir' and
                   (np.round(max_Ir_Z - atom.z, 4) == 0.0953) and 
                    atom.index not in Ir_bri)])
x_grid = list(set([pos[0] for pos in Ir_cus]))
y_grid = list(set([pos[1] for pos in Ir_cus]))
x_grid_m = list(set([pos[0] for pos in Ir_bri]))
y_grid_m = list(set([pos[1] for pos in Ir_bri]))
x_grid.sort()
y_grid.sort()
x_grid_m.sort()
y_grid_m.sort()
print(np.shape(positions))
# %%
def plot_movement(positions, init_struc, max_Ir_Z):
    # fig = plt.figure()
    cm  = plt.get_cmap('gist_rainbow')
    fig, (ax1, ax2)  = plt.subplots(2,1)
    ax1.set_xlim(0, init_struc.cell[0][0])
    ax2.set_xlim(0, init_struc.cell[0][0])
    ax1.set_ylim(0, init_struc.cell[1][1])
    ax1.set(xlabel='$x$', ylabel='$y$')
    ax2.set(xlabel='$x$', ylabel='$z$')
    ax1.set_xticks(x_grid)
    ax2.set_xticks(x_grid)
    ax1.set_yticks(y_grid)
    ax1.set_xticks(x_grid_m, minor=True)
    ax1.set_yticks(y_grid_m, minor=True)
    ax1.grid(which='major')
    ax1.grid(which='minor', alpha=0.5, linestyle='--')
    for ind1, pos_set in enumerate(positions):
        ax1.set_prop_cycle(color=[cm(1.*i/len(pos_set)) for i in range(len(pos_set))])
        ax2.set_prop_cycle(color=[cm(1.*i/len(pos_set)) for i in range(len(pos_set))])
        if ind1==0: # setting color labels only first iter cos rest are the same
            for ind2, pos_H in enumerate(pos_set):
                ax1.plot(pos_H[0], pos_H[1], 'o', markersize=1) #label=f'{ind2}')
                ax2.plot(pos_H[0], pos_H[2]-max_Ir_Z, 'o', markersize=1) #label=f'{ind2}')    
        else:
            for ind2, pos_H in enumerate(pos_set):
                ax1.plot(pos_H[0], pos_H[1], 'o', markersize=1)
                ax2.plot(pos_H[0], pos_H[2]-max_Ir_Z, 'o', markersize=1)    
    # ax1.legend(loc='lower center',prop={'size': 18}, bbox_to_anchor='tight')
    # ax2.legend(loc='lower center',prop={'size': 18}, bbox_to_anchor='tight')
    fig.set_size_inches(8,12)
    plt.savefig("H_movement_grid.png", dpi=300, bbox_inches='tight')

plot_movement(positions, init_struc, max_Ir_Z)
# %%
