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
def plot_free_eners_by_index(csv_files, skip_init=10, save_fig=True, save_fig_path=None):
    fig, axs = plt.subplots(1,1)
    fig_fake, axs_fake = plt.subplots()
    fig_fake.set_visible(False)
    dfs = []
    colors = pl.cm.jet(np.linspace(0, 1, 1+len(csv_files)))
    for ind, file in enumerate(csv_files):
        df   = pd.read_csv(file, index_col=0)[skip_init:]
        dfs.append(df)
        density = stats.gaussian_kde(df['Free_Energy'])
        n, x, _ = axs_fake.hist(df['Free_Energy'], bins=50, color=colors[ind],  
                                histtype=u'step', density=True, label=ind)  
        plt.clf()  
        axs.plot(x, density(x), label=ind, color=colors[ind])
    fig.set_size_inches(10,6)
    leg = axs.legend(loc='center left', prop={'size': 15},  bbox_to_anchor=(1, 0.5), ncols=1)
    leg.set_title('Index')
    axs.set(xlabel='$Internal$ $Energy$ $(in$ $eV)$', ylabel='$Frequency$')
    if save_fig:
        fig.savefig(f'{save_fig_path}/free_ener_freq.png')
    return dfs
#%%
def plot_free_eners_by_stoich(csv_files, atom_type='O', skip_init=10, cmap=pl.cm.winter, save_fig=False, save_fig_path=None):
    dfs = []
    for file in csv_files:
        dfs.append(pd.read_csv(file, index_col=0)[skip_init:])
    df_joint = pd.concat(dfs)
    df_joint['Tot_O'] = 2*df_joint['OO'] + df_joint['OH'] + df_joint['O'] + df_joint['OH2']
    df_joint['Tot_H'] = 2*df_joint['OH2'] + df_joint['OH'] + df_joint['OO']
    if atom_type=='O':
        max_min_diff = df_joint['Tot_O'].max() - df_joint['Tot_O'].min()
        min_val = df_joint['Tot_O'].min()
    if atom_type=='H':
        max_min_diff = df_joint['Tot_H'].max() - df_joint['Tot_H'].min()
        min_val = df_joint['Tot_H'].min()
        print(df_joint['Tot_H'].max())
    colors = cmap(np.linspace(0, 1, 1 + max_min_diff))
    fig, axs = plt.subplots(1,1)
    fig_fake, axs_fake = plt.subplots()
    for ind, file in enumerate(csv_files):
        df   = pd.read_csv(file, index_col=0)[10:]
        df['Tot_O'] = 2*df['OO'] + df['OH'] + df['O'] + df['OH2']
        df['Tot_H'] = df['OH']   + 2*df['OH2']
        density = stats.gaussian_kde(df['Free_Energy'])
        n, x, _ = axs_fake.hist(df['Free_Energy'], bins=50, color=colors[df[f'Tot_{atom_type}'].iloc[0]-min_val],
                                 histtype=u'step', density=True, label=df[f'Tot_{atom_type}'].iloc[0])
        plt.clf()
        axs.plot(x, density(x), label=df[f'Tot_{atom_type}'].iloc[0], color=colors[df[f'Tot_{atom_type}'].iloc[0]-min_val])
    axs.set(xlabel='$Internal$ $Energy$ $(in$ $eV)$', ylabel='$Frequency$')
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size = "5%", pad = 0.25)
    fig.add_axes(cax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(min_val, max_min_diff+min_val))                  
    fig.colorbar(mappable=sm, cax=cax,
                ticks = np.linspace(min_val, max_min_diff+min_val, 1+max_min_diff),
                orientation = 'vertical', label=f'$n_{atom_type}$'
            )
    fig.set_size_inches(10,6)
    if save_fig:
        fig.savefig(f'{save_fig_path}/free_ener_by_stoich.png')
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
            plt.savefig(f'{save_fig_path}/{site}_{use_x}.png')

def plot_trends_water(csv_files, save_fig=False, save_fig_path=None, skip_init=10, color='lightsteelblue'):
    dfs = []
    for file in csv_files:
        dfs.append(pd.read_csv(file, index_col=0)[skip_init:])
    df_joint= pd.concat(dfs)
    print(len(df_joint))
    cats = ['H2O', 'OH']
    for use_x in cats:
        fig, axs = plt.subplots(2,1)
        sns.violinplot(x=use_x, y='Free_Energy', data=df_joint, color=color, ax = axs[0])
        axs[0].set(ylabel='$Internal$ $Energy$ $(in$ $eV)$')
        df_joint[use_x].value_counts().sort_index().plot(kind='bar', ax=axs[1], color=color)
        axs[1].set(xlabel=f'$n$$_{{{use_x}}}$', ylabel='$Count$')
        fig.set_size_inches(10,8.5)
        plt.subplots_adjust(hspace=0.0, wspace=0.0)
        if save_fig:
            plt.savefig(f'{save_fig_path}/water_{use_x}.png')
# %%
def main():
    plot_path = Path.cwd()/'TRENDs'
    plot_path.mkdir(parents=True, exist_ok=True)

    #surf_csv_files = glob.glob("./CSVs/cus*.csv")
    #dataframes= plot_free_eners_by_index(csv_files=surf_csv_files, save_fig=True, save_fig_path=plot_path)
    #plot_free_eners_by_stoich(csv_files=surf_csv_files, save_fig=True, save_fig_path=plot_path)
    #plot_trends(dfs=dataframes, site='cus', save_fig=True, save_fig_path=plot_path)

    water_csv_files = glob.glob("./CSVs/water_df_*.csv")
    plot_trends_water(csv_files=water_csv_files, save_fig=True, save_fig_path=plot_path)

if __name__=='__main__':
    main()
# %%