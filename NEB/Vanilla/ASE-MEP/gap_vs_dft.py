from ase.io import read, write
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
font = {'size'   : 16}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)

import os
from pathlib import Path
cwd = Path(os.getcwd())
path1 = cwd/'mep_latest.xyz'
path2 = cwd/'mep_gap.xyz'
path3 = cwd/'mep_mace.xyz'
strucs1 = read(path1, ":")
strucs2 = read(path2, ":")
strucs3 = read(path3, ":")

e_zero_1 = strucs1[0].info['energy']
e_zero_2 = strucs2[0].info['GAP_energy']
e_zero_3 = strucs3[0].info['MACE_energy']

dft_E  = [im.info['energy']-e_zero_1 for im in strucs1]
gap_E  = [im.info['GAP_energy']-e_zero_2 for im in strucs2]
mace_E = [im.info['MACE_energy']-e_zero_3 for im in strucs3]

plt.plot(np.arange(0,len(strucs1)), dft_E, marker="o", label="DFT")
plt.plot(np.arange(0,len(strucs1)), gap_E, marker="o", label="GAP")
plt.plot(np.arange(0,len(strucs1)), mace_E, marker="o", label="MACE")

plt.xlabel("Reaction coordinate")
plt.ylabel("Relative energy, eV")
plt.grid()
plt.legend()
plt.show()
