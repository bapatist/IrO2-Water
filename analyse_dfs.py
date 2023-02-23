#%%
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.pylab as pl
font = {'size'   : 18}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
import seaborn as sns
#%%
def plot_free_eners(csv_files, save_fig=True, save_fig_path=None):
    fig, axs = plt.subplots(1,1)
    fig_fake, axs_fake = plt.subplots()
    fig_fake.set_visible(False)
    dfs = []
    colors = pl.cm.jet(np.linspace(0,1,11))
    for ind, file in enumerate(csv_files):
        df   = pd.read_csv(file, index_col=0)[10:]
        dfs.append(df)
        density = stats.gaussian_kde(df['Free_Energy'])
        n, x, _ = axs_fake.hist(df['Free_Energy'], bins=50, color=colors[ind],  
                                histtype=u'step', density=True, label=ind)  
        plt.clf()  
        axs.plot(x, density(x), label=ind, color=colors[ind])
    fig.set_size_inches(10,6)
    leg = axs.legend(loc='center right', prop={'size': 15})
    leg.set_title('Index')
    axs.set(xlabel='$Internal$ $Energy$ $(in$ $eV)$', ylabel='$Frequency$')
    if save_fig:
        fig.savefig(f'{save_fig_path}/free_ener_freq.png')
    return dfs
#%%
def plot_trends(dfs, site='cus', 
                color='lightsteelblue', 
                save_fig=True, save_fig_path=None):
    df_joint = pd.concat(dfs)
    if site=='cus':
        site_ids = ['OO', 'O', 'OH', 'OH2']
    elif site=='bri':
        site_ids = ['O', 'OH']
    for use_x in site_ids:
        fig, axs = plt.subplots(2,1)#, sharex='col', sharey='row')
        #hb = axs[0].hexbin(x_ax, y_ax, gridsize=25, mincnt=1, bins = 'log', cmap='GnBu')
        sns.violinplot(x=use_x, y='Free_Energy', data=df_joint, color=color, ax = axs[0])
        # bin_means, bin_edges, binnumber = stats.binned_statistic(x_ax, y_ax, statistic='median', bins=2+x_ax.max()-x_ax.min(), range=(x_ax.min()-1,x_ax.max()+1))
        # axs.hlines(bin_means, bin_edges[:-1]-0.5, bin_edges[1:]-0.5, colors='k', lw=1,label='$Binned$ $Average$')
        axs[0].set(ylabel='$Internal$ $Energy$ $(in$ $eV)$')

        df_joint[use_x].value_counts().sort_index().plot(kind='bar', ax=axs[1], color=color)
        axs[1].set(xlabel=f'$n$$_{{{use_x}}}$', ylabel='$Count$')
        fig.set_size_inches(10,8.5)
        plt.subplots_adjust(hspace=0.0, wspace=0.0)
        if save_fig:
            plt.savefig(f'{save_fig_path}/{use_x}.png')
# %%
def main():
    csv_files = glob.glob("./CSVs/cus*.csv")
    plot_path = Path.cwd()/'TRENDS'  
    plot_path.mkdir(parents=True, exist_ok=True)
    dataframes= plot_free_eners(csv_files=csv_files, save_fig=True, save_fig_path=plot_path)
    plot_trends(dfs=dataframes, site='cus', save_fig=True, save_fig_path=plot_path)

if __name__=='__main__':
    main()
# %%
