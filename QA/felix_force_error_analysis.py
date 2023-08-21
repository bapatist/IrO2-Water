#%%
from ase.io import read
import numpy as np
import sys, os
from matplotlib import pyplot as plt
sys.path.append(os.getcwd())
#%%
def f_rmse(path, cut=0.4, avr=None, plot='Norm', save=None):
    """
    Parameters
    ----------
    path: str,
        file_path
    cut: float,
        lower limit of percentage area [eV/AA]
    avr: int,
        amount of points to average over
    plot: str,
        Norm or Component
    save: str
        Filepath for saving figure, if None figure will not be saved.

    Returns
    -------
        None
    """

    atoms = read(path, ':')
    percent_list = []
    low_forces = []
    DFT_forces = []
    low_force_component = []
    DFT_force_component = []
    for at in atoms:
        if at.info['config_type'] != 'isolated_atom' and at.info['config_type'] != "dimer":
            DFT = np.linalg.norm(at.arrays['DFT_forces'], axis=1)
            calc = np.linalg.norm(at.arrays['GAP_forces'], axis=1)
            DFT_component = at.arrays['DFT_forces'].flatten()
            calc_component = at.arrays['GAP_forces'].flatten()
            percent = (abs(DFT - calc) / np.where(DFT < cut, cut, DFT)) * 100
            for i, val in enumerate(DFT):
                if val <= cut:
                    low_forces.append(abs(DFT[i] - calc[i]))
                    DFT_forces.append(DFT[i])
                else:
                    percent_list.append(percent[i])
            low_force_component = [*low_force_component, *list(abs(calc_component - DFT_component))]
            DFT_force_component = [*DFT_force_component, *list(abs(DFT_component))]

    print(np.mean(percent_list))

    if plot == 'Norm':
        list1, list2 = zip(*sorted(zip(DFT_forces, low_forces)))
    elif plot == 'Component':
        list1, list2 = zip(*sorted(zip(DFT_force_component, low_force_component)))
    else:
        print('no plot possible')
        sys.exit()

    if type(avr) == int:
        count = 0
        x_axis = [list1[0]]
        p_x_axis = []
        full = []
        part = []
        pc = []
        ppc = []
        for i in range(len(list1)):
            if list1[i] < 0.01:
                if count < avr:
                    part.append(list2[i])
                    count += 1
                else:
                    full.append(np.mean(part))
                    part = [list2[i]]
                    x_axis.append((list1[i]))
                    count = 0
            else:
                if len(p_x_axis) == 0:
                    p_x_axis.append(list1[i])
                if count < avr:
                    part.append(list2[i])
                    ppc.append(list2[i] / list1[i])
                    count += 1
                else:
                    full.append(np.mean(part))
                    pc.append(np.mean(ppc))
                    part = [list2[i]]
                    ppc = [list2[i] / list1[i]]
                    p_x_axis.append(list1[i])
                    x_axis.append(list1[i])
                    count = 0

        part.append(list2[-1])
        full.append(np.mean(part))
        full.append(5)
        x_axis.append(5)

        ppc.append(list2[-1] / list1[-1])
        pc.append(np.mean(part))
        pc.append(5)
        p_x_axis.append(5)

        fig, ax = plt.subplots()
        ax.step(np.array(x_axis) * 1000, np.array(full) * 1000, 'k', where='post')
        ax.legend([f'MAE over {avr} points'])
        ax.scatter(np.array(list1) * 1000, np.array(list2) * 1000, alpha=0.1, s=3)
        ax.set_ylim([0, cut*1000])
        ax.set_xlim([0, cut*1000])
        ax.set_xlabel(f'DFT Force ({plot}) ' + "$[\mathrm{mev/\AA}]$")
        ax.set_ylabel(f'Absolut Error ({plot}) ' "$[\mathrm{mev/\AA}]$")

        secax = ax.twinx()
        secax.step(np.array(p_x_axis) * 1000, np.array(pc) * 100, 'r', where='post')
        secax.set_ylabel('Force [%]', color='r')
        secax.tick_params(axis='y', labelcolor='r')
        secax.set_ylim([0, 100])
        if type(save) == str:
            plt.savefig(save, dpi=300)

        plt.show()
#%%
def main():
    f_rmse(path="./SAMPLES/ts_with_GAP_0.001sig.xyz", 
           plot="Component",
           cut=0.4,
           avr =1000,
           save="felix.png")

if __name__ == "__main__":
    main()
# %%
