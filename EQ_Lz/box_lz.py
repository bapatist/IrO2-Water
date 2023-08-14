#%%
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib

font = {'size'   : 16}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
#%%
log = pd.read_table("./logCut.lammps", delim_whitespace=True)
print(log.columns)
fig, ax = plt.subplots()
ax.plot(log['Time'], log['Lz'])
ax.set(xlabel='$\mathrm {time,\ ps}$', ylabel='$\mathrm {L_z,\ \AA}$')
ax.yaxis.labelpad = 10
plt.savefig("lz.png", dpi=300, bbox_inches='tight')
# %%
