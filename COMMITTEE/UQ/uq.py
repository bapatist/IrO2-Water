#%%
from ase.io import read
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
font = {'size'   : 18}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
#%%
test_set = read("./testing_ener_calib.xyz", ":")

DFT_ener  = [im.info['DFT_energy'] for im in test_set]
COMM_ener = [im.info['committee_energy'] for im in test_set]
fig, ax0 = plt.subplots()
ax0.scatter(DFT_ener, COMM_ener, s=5, c='k')
lims = [
    np.min([ax0.get_xlim(), ax0.get_ylim()]),  # min of both axes
    np.max([ax0.get_xlim(), ax0.get_ylim()]),  # max of both axes
]
ax0.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax0.set_xlim(lims)
ax0.set_ylim(lims)
ax0.set(xlabel="DFT energy, eV", ylabel="committe-GAP energy, eV")
del_E = [im.info['DFT_energy']-im.info['committee_energy'] for im in test_set]
uncer_E = [1.96*im.info['committee_energy_uncertainty'] for im in test_set]
fig.savefig('scatter_energy.png', bbox_inches='tight', dpi=300)

#%%
fig, ax1 = plt.subplots()
ax1.plot(np.arange(len(test_set)), del_E, color='k', alpha=1.0, lw=0.75, label='Absolute Error')
# ax1.plot(np.arange(len(test_set)), uncer, color='b', alpha=0.75, lw=0.75, label='95% Confidence interval')
ax1.fill_between(np.arange(len(test_set)), [-i for i in uncer_E], uncer_E, 
                 color='k', alpha=0.15, label='95\% confidence interval')
ax1.legend()
ax1.set(xlabel='Index', ylabel='Error (and uncertainty)')
fig.set_size_inches(10,4)
fig.savefig('confidence.png', bbox_inches='tight', dpi=300)

#%%
fig, ax2 = plt.subplots()
ax2.scatter(uncer_E, del_E, s=10)
lims = [
    np.min([ax2.get_xlim(), ax2.get_ylim()]),  # min of both axes
    np.max([ax2.get_xlim(), ax2.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax2.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax2.set_aspect('equal')
ax2.set_xlim(lims)
ax2.set_ylim(lims)
ax2.set(xlabel='Uncertainty', ylabel='Absolute Error')
# fig.savefig('/Users/paul/Desktop/so.png', dpi=300)
# %%
DFT_forces  = np.hstack([np.hstack(im.arrays['DFT_forces']) for im in test_set])
COMM_forces = np.hstack([np.hstack(im.arrays['committee_forces']) for im in test_set])
fig, ax3 = plt.subplots()
ax3.scatter(DFT_forces, COMM_forces, s=10)
# %%
