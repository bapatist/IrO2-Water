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
    df   = pd.read_csv(file)[20:]
    dfs.append(df)
    axs[0].hist(df['Free_Energy'], bins=50, label=ind)
    axs[1].hist(df['OH'], label=ind)
    axs[2].hist(df['OO'], label=ind)
axs[1].legend()
#%%
df_joint = pd.concat(dfs)
fig, axs = plt.subplots()
axs.plot(df_joint['OO'], df_joint['Free_Energy'], ls='None', marker='o')

# %%
