#%%
import pandas as pd
import scipy.stats as stats
import glob
from matplotlib import pyplot as plt
#%%
csv_files = glob.glob("./CSVs/cus*.csv")
fig, axs = plt.subplots(3,1)
dfs = []
for ind, file in enumerate(csv_files):
    df   = pd.read_csv(file)[2:]
    dfs.append(df)
    axs[0].hist(df['Free_Energy'], bins=50, label=ind)
    axs[1].hist(df['OH'], label=ind)
    axs[2].hist(df['OO'], label=ind)
axs[1].legend()
#%%
df_joint = pd.concat(dfs)
fig, axs = plt.subplots()
x_ax = df_joint['OO']
y_ax = df_joint['Free_Energy']
# axs.plot(x_ax, y_ax, ls='None', marker='o')
hb = axs.hexbin(x_ax, y_ax, gridsize=20, mincnt=1, bins = 'log', cmap='GnBu')

bin_means, bin_edges, binnumber = stats.binned_statistic(x_ax, y_ax, statistic='mean', bins=2+x_ax.max()-x_ax.min(), range=(x_ax.min()-1,x_ax.max()+1))
axs.hlines(bin_means, bin_edges[:-1]-0.5, bin_edges[1:]-0.5, colors='k', lw=4,label='$Binned$ $Average$')

# %%
