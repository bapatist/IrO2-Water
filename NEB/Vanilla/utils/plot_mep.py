from ase.io import read, write
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
font = {'size'   : 16}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)

a2b = read("../ASE-MEP/A2B_temp.xyz", ":")
N_images = 7
e_string = 'energy'

N_paths  = int(len(a2b)/N_images)
start_E  = a2b[0].info[e_string]
cmap = plt.get_cmap('viridis')
len(a2b)
for i in range(N_paths):
    clr = cmap(i / (N_paths - 1))
    x, y = [], []
    for j in range(i*N_images, (i+1)*N_images, 1):
        x.append(j%N_images)
        y.append(a2b[j].info[e_string]-start_E)
    plt.plot(x,y, marker = "o", color=clr, label=str(i))
plt.xticks(np.arange(0,N_images))
plt.xlabel("$\mathrm {Reaction\ coordinate}$")
plt.ylabel("$\mathrm {Relative\ energy,\ eV}$")
plt.grid()
plt.show()

# Save last MEP
# write("./MEP/mep_latest.xyz", a2b[-N_images:])