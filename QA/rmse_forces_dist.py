#%%
import numpy as np
import scipy.stats as stats
from ase.io import read
from matplotlib import pyplot as plt
import matplotlib
font = {'size': 18}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
#%%
def _perc_err(ground_truth, predictions, low_cap):
    absolute_error = np.abs(predictions-ground_truth)
    if abs(ground_truth)<low_cap:
        ground_truth = low_cap
    percentage_error = (absolute_error / np.abs(ground_truth)) * 100
    return absolute_error, percentage_error

def main(ts, gap_or_mace, up_cap, low_cap, nbins, save):
    # up_cap is used to calc and plot only till up_cap
    # low_cap is used to calc and plot only till low_cap
    # returns DFT and % errors
    dft_vals, p_errors, abs_errors = [], [], []
    for im in ts:
        dft_forces = np.hstack(im.arrays['DFT_forces'])
        gap_forces = np.hstack(im.arrays[f'{gap_or_mace}_forces'])
        for dft, gap in zip(dft_forces, gap_forces):
            abs_err, perc_err = _perc_err(ground_truth=dft, 
                                          predictions=gap, 
                                          low_cap=low_cap)
            abs_errors.append(abs_err)
            p_errors.append(perc_err)
            dft_vals.append(dft)
    x, y1, y2 = [], [], []
    for d, p, a in zip(dft_vals, p_errors, abs_errors):
        x.append(abs(d))
        y1.append(p)
        y2.append(a)
    bin_means_p, bin_edges_p, binnumber_p = stats.binned_statistic(x, 
                                                            y1, 
                                                            statistic='mean', 
                                                            bins=nbins)
    bin_means_a, bin_edges_a, binnumber_a = stats.binned_statistic(x, 
                                                            y2, 
                                                            statistic='mean', 
                                                            bins=nbins)
    fig, ax = plt.subplots()
    ax.scatter(x, y2, s=5, alpha=0.1, marker="o", color='gray', zorder=2,
                       label="$\mathrm {Abs.\ error}$")
    ax.step(bin_edges_a[:-1], bin_means_a, 'k', where='post', 
                    label="$\mathrm {Mean\ abs.\ error}$")
    ax.set_ylim(0, up_cap)
    ax.set_xlim(low_cap, up_cap)
    ax.set(xlabel="$\mathrm {DFT_{(F_x,\ F_y,\ F_z)},\ eV/\AA}$",
           ylabel="$\mathrm{Absolute\ error,\ eV/\AA}$")
    secax = ax.twinx()
    # dots2 = ax.scatter(x, y1, alpha=0.01, marker="o", color='orange',
    #                       label="$\mathrm {\%\ error}$")
    secax.step(bin_edges_p[:-1], bin_means_p, c='r', where='post', 
               label="$\mathrm {Mean\ \%\ error}$")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = secax.get_legend_handles_labels()
    leg = secax.legend(lines + lines2, labels + labels2, loc=0)
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    secax.set_ylabel("$\mathrm{\%\ Error}$", color='r')
    secax.tick_params(axis='y', labelcolor='r')
    secax.set_ylim([0, 100])
    plt.title(f"{gap_or_mace}")
    plt.savefig(save, dpi=300, bbox_inches='tight')
if __name__ == "__main__":
    ts = read("MACE_TRAIN+TEST/test_all_I4_MACE_GPU.xyz", ":")
    gap_or_mace = "MACE"
    # ts = read("TRAIN_SET/I4/ts_lowF.xyz", ":")
    # gap_or_mace = "GAP"
    main(ts, gap_or_mace=gap_or_mace, 
         up_cap=0.5, low_cap=0.001, nbins=7500,
         save=f"./PLOTs/{gap_or_mace}_I4_lowF_force_steps.png")
# %%
