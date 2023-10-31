#%%
import numpy as np
import pandas as pd
from ase.io import read, write
from matplotlib import pyplot as plt
import matplotlib
font = {'size': 18}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
#%%
mdA = read("./eval_outs/md_eval_A_fixed.xyz", ":")
mdB = read("./eval_outs/md_eval_B_fixed.xyz", ":")
mdC = read("./eval_outs/md_eval_C_fixed.xyz", ":")

assert len(mdA)==len(mdB)==len(mdC)
n_images = len(mdA)
n_atoms = len(mdA[0])
#%%
def sample_by_energy(mdA, mdB, mdC, n_atoms, n_samples):
    means, std_devs = [], []
    times = []
    for i, (a,b,c) in enumerate(zip(mdA, mdB, mdC)):
        energies = [a.info['MACE_energy']/n_atoms, b.info['MACE_energy']/n_atoms, c.info['MACE_energy']/n_atoms]
        std_devs.append(np.std(energies))
        means.append(np.mean(energies))
        # times.append(i)
        times.append(a.info['time'])
    std_devs = np.array(std_devs)
    samp_ids = std_devs.argsort()[-n_samples:][::-1]
    return times, means, std_devs, samp_ids

t_arr, mean_Es, std_Es, samp_IDs = sample_by_energy(mdA=mdA, mdB=mdB, mdC=mdC, 
                                                    n_atoms=n_atoms, 
                                                    n_samples=10)
plt.plot(t_arr, mean_Es, lw=0.5)
plt.xlabel("$\mathrm{Time,\ ps}$")
plt.ylabel("$\mathrm{\mu_{energy-per-atom}\ ,\ eV}$")
plt.savefig("PLOTs/mean_energy_vs_time.png",dpi=300, bbox_inches='tight')
plt.show()
plt.plot(t_arr, std_Es, lw=0.5)
plt.xlabel("$\mathrm{Time,\ ps}$")
plt.ylabel("$\mathrm{\sigma_{energy-per-atom}\ ,\ eV}$")
plt.savefig("PLOTs/mean_std_vs_time.png",dpi=300, bbox_inches='tight')
plt.show()

plt.hist(std_Es, bins=100, color='gray')  # You can adjust the number of bins as needed
# plt.yscale('log') 
plt.xlabel('$\mathrm{\sigma_{energy\ per-atom}\ ,\ eV}$')
plt.ylabel('$\mathrm{frequency}$')
plt.savefig("PLOTs/std_dev_energy_hist.png" ,dpi=300, bbox_inches='tight')
print(samp_IDs)
#%%
def sample_by_force(mdA, mdB, mdC, n_atoms, n_samples):
    std_devs = []
    for i, (a,b,c) in enumerate(zip(mdA, mdB, mdC)):
        a_f = np.array(a.arrays['MACE_forces'].flatten())
        b_f = np.array(b.arrays['MACE_forces'].flatten())
        c_f = np.array(c.arrays['MACE_forces'].flatten())
        for fa, fb, fc in zip(a_f, b_f, c_f):
            sigma = np.std([fa, fb, fc])
            std_devs.append({'ID':i, 'std':sigma})
    return std_devs
std_Fs = sample_by_force(mdA=mdA, mdB=mdB, mdC=mdC, 
                        n_atoms=n_atoms, 
                        n_samples=10)
df = pd.DataFrame(std_Fs)
#%%
print("Mean value of std", df['std'].mean(), "eV/Ang.")
print("Top 10 rows with highest std \n", df.sort_values(by=['std'], ascending=False)[:10])
plt.hist(df['std'], bins=1000, color='gray')  # You can adjust the number of bins as needed
plt.yscale('log') 
plt.xlabel('$\mathrm{\sigma_{forces}\ ,\ eV/\AA}$')
plt.ylabel('$\mathrm{frequency}$')
plt.savefig("PLOTs/std_dev_forces_hist.png" ,dpi=300, bbox_inches='tight')
filtered_df = df[df['std'] > 0.5]
print(len(filtered_df), "atoms found with >0.5 eV/ang std")
unique_id_count = filtered_df['ID'].nunique()
print( "from", unique_id_count, "unique md-images")
#%% write xyz for images with highes forces std
df_order = df.sort_values(by=['std'], ascending=False)
id_top20 = pd.unique(df_order['ID'])[:20]
im_top20 = [mdA[i] for i in id_top20]
write("top20.xyz", im_top20)
# %%
