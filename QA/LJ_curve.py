# %%
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
font = {'size'   : 18}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
#%%
from ase.io import read
# %%
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

ts = read("./SAMPLES/ts_with_GAP_0.001sig.xyz", ":")
dims = [s for s in ts if s.info['config_type']=="dimer"]
# %%
dists, gap_E, dft_E = [], [], []
for im in dims:
    if im.get_chemical_formula() == "O2":
        dists.append(im.get_distance(0,1))
        gap_E.append(im.info['GAP_energy'])
        dft_E.append(im.info['DFT_energy'])

fig, ax = plt.subplots()
ax.plot(dists, dft_E, alpha=0.5, marker=".", c='k', label="DFT")
ax.plot(dists, gap_E, alpha=0.5, ls="--", marker=".", label="GAP")
ax.set(xlabel="$\mathrm {distance,\ \AA}$", ylabel="$\mathrm {Energy,\ eV}$", 
       title="$\mathrm {O-O\ rigid\ scan}$")
ax.legend()
plt.savefig("PLOTs/o2_LJ.png", dpi=300, bbox_inches='tight')
# %%
rms_ener(dft_E, gap_E)
# %%
