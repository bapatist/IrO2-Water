#%%
import numpy as np
import pandas as pd
from ase.io import read, write
from matplotlib import pyplot as plt
import matplotlib
font = {'size': 18}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
from scipy import stats  # Add this import statement
#%%
strucs = read("./eval_outs_trainS/trainS_MACE_all_and_ABC.xyz", ":")
strucs = [im for im in strucs if im.info['config_type'][:4]=='surf']
n_images = len(strucs)
n_atoms = len(strucs[0])
#%%
def plot_scatter(x, y, xlabel):
    fig, ax = plt.subplots()
    hb = ax.hexbin(x, y, gridsize=100, bins='log', mincnt=1, cmap='viridis')
    # Add a colorbar
    cbar = plt.colorbar(hb)
    cbar.set_label('$\mathrm {Density}$')
    ax.set(xlabel=xlabel, ylabel="$\mathrm{Absolute\ error\ ,\ eV/\AA}$")
    plt.savefig("PLOTs/trainSet_bench_Fnorm.png" ,dpi=300, bbox_inches='tight')

def plot_scatter_with_mean_std_binned(x, y, xlabel, num_bins):
    fig, ax = plt.subplots()
    hb = ax.hexbin(x, y, gridsize=100, bins='log', mincnt=1, cmap='viridis')

    # Bin the x values into equidistant slices and calculate mean and standard deviation for each bin
    bin_means, bin_edges, _ = stats.binned_statistic(x, y, statistic='mean', bins=num_bins)
    bin_std, _, _ = stats.binned_statistic(x, y, statistic='std', bins=num_bins)

    # Calculate the bin centers
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # Plot the mean values as a line
    ax.plot(bin_centers, bin_means, color='red', label='Mean')

    # Fill between the mean +/- standard deviation
    ax.fill_between(bin_centers, bin_means - bin_std, bin_means + bin_std, color='red', alpha=0.2)

    # Add a colorbar
    cbar = plt.colorbar(hb)
    cbar.set_label('$\mathrm {Density}$')

    ax.set(xlabel=xlabel, ylabel="$\mathrm{Absolute\ error\ ,\ eV/\AA}$")
    plt.legend()
    plt.savefig("PLOTs/trainSet_bench_Fcomp_avgOFfitABC.png", dpi=300, bbox_inches='tight')
    plt.show()

# Example usage:
# Replace x, y, xlabel, and num_bins with your data and desired parameters
# plot_scatter_with_mean_std_binned(x, y, xlabel, num_bins)

def get_std_forces(strucs, std_by_type):
    all_info = []
    for i, im in enumerate(strucs):
        if std_by_type == "component":
            AE_A = abs(np.array(im.arrays['MACE_fitA_forces'].flatten()) - 
                     np.array(im.arrays['DFT_forces'].flatten()))
            AE_B = abs(np.array(im.arrays['MACE_fitB_forces'].flatten()) - 
                     np.array(im.arrays['DFT_forces'].flatten()))
            AE_C = abs(np.array(im.arrays['MACE_fitC_forces'].flatten()) - 
                     np.array(im.arrays['DFT_forces'].flatten()))
            AE   = np.mean([AE_A, AE_B, AE_C], axis=0)
            a_f  = np.array(im.arrays['MACE_fitA_forces'].flatten())
            b_f  = np.array(im.arrays['MACE_fitB_forces'].flatten())
            c_f  = np.array(im.arrays['MACE_fitC_forces'].flatten())
        if std_by_type == "norm":
            print("WARNING: fitA, B , C average not implemented here. Showing versus abs error of big mace fit")
            AE = abs(np.linalg.norm(im.arrays['MACE_forces'], axis=1) - 
                     np.linalg.norm(im.arrays['DFT_forces'], axis=1))
            a_f = np.linalg.norm(im.arrays['MACE_fitA_forces'], axis=1)
            b_f = np.linalg.norm(im.arrays['MACE_fitB_forces'], axis=1)
            c_f = np.linalg.norm(im.arrays['MACE_fitC_forces'], axis=1)
        for fa, fb, fc, ae  in zip(a_f, b_f, c_f, AE):
            sigma = np.std([fa, fb, fc])
            all_info.append({'ID':i, 'std':sigma, 'ae':ae, 
                            'config_type':im.info['config_type']})
    df = pd.DataFrame(all_info)
    plot_scatter_with_mean_std_binned(x=df['std'],y=df['ae'], 
                                      xlabel="$\mathrm{\sigma_{forces}\ ,\ eV/\AA}$",
                                      num_bins=50)
    return df

def get_std_energy(strucs):
    all_info = []
    for i, im in enumerate(strucs):
        ae = 1000*abs(im.info['DFT_energy']-im.info['MACE_energy'])/len(im)
        abc_energies = [im.info['MACE_fitA_energy'], im.info['MACE_fitB_energy'], im.info['MACE_fitC_energy']]
        std = 1000*np.std(abc_energies)/len(im)
        all_info.append({'ID':i, 'std':std, 'ae':ae, 
                         'config_type':im.info['config_type']})
    df = pd.DataFrame(all_info)
    plt.scatter(df['std'], df['ae'])
    plt.xlabel("$\mathrm{\sigma_{energy},\ meV}$")
    plt.ylabel("$\mathrm{Absolute\ error\ ,\ meV}$")
    # plt.savefig("PLOTs/trainS_bench_energy.png" ,dpi=300, bbox_inches='tight')
    return df
#%%
df = get_std_forces(strucs, std_by_type="component")
# get_std_energy(strucs)
# %% Plot histogram of STD for forces
print("Mean value of std", df['std'].mean(), "eV/Ang.")
print("Top 10 rows with highest std \n", df.sort_values(by=['std'], ascending=False)[:10])
plt.hist(df['std'], bins=1000, color='gray')  # You can adjust the number of bins as needed
plt.yscale('log') 
plt.xlabel('$\mathrm{\sigma_{forces}\ ,\ eV/\AA}$')
plt.ylabel('$\mathrm{frequency}$')
# plt.savefig("PLOTs/std_dev_forces_hist.png" ,dpi=300, bbox_inches='tight')
filtered_df = df[df['std'] > 0.5]
print(len(filtered_df), "atoms found with >0.5 eV/ang std")
unique_id_count = filtered_df['ID'].nunique()
print( "from", unique_id_count, "unique md-images")
# %%
