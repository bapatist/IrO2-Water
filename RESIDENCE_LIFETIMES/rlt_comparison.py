# %%
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
font = {'size'   : 18}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
# %%
O_cov  = [54.73,  19.98, 23.06]
OH_cov = [145.99, 42.00, 25.80]
labels = ['Layer I', 'Layer II', 'Bulk']
#%%
fig, ax = plt.subplots()
ax.bar()
#%%
coverage = ('$\mathrm {OH\ covered}$', '$\mathrm {O\ covered}$')
rlt = {
    '$\mathrm {Layer\ I}$': (145.99, 54.73),
    '$\mathrm {Layer\ II}$': (42.00, 19.98),
    '$\mathrm {Bulk\ region}$': (25.80, 23.06),
}
colors = ['cornflowerblue', 'blue', 'midnightblue']

x = np.arange(len(coverage))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0.0

fig, ax = plt.subplots(layout='constrained')

for i, (attribute, measurement) in enumerate(rlt.items()):
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    # ax.bar_label(rects, padding=3)
    rects[0].set_color(colors[i])
    rects[1].set_color(colors[i])
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('$\mathrm {Lifetime\ \\tau,\ ps}$')
# ax.set_title('$\mathrm {}$')
ax.set_xticks(x + width, coverage)
ax.legend()
fig.set_size_inches(5,5)
plt.savefig("compare_bar_plot.png", dpi=300, bbox_inches='tight')
# %%