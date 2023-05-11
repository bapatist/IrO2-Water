#%%
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
font = {'size'   : 18}
#matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
# %%
@dataclass
class TrajProcessor():
    sim_indices: list # two numbers: init and final index
    sim_paths: list
    init_struc_paths: list
    skip_n_frames: int
    iridium_surf_Zs: list #top and bottom surface iridium Z

    def __post_init__(self) -> None:
        """Reads all trajectories and init_strucs"""
        from ase.io.trajectory import Trajectory
        from ase.io import read
        _trajs, _mixes = [], []
        for _path, _mix in zip(self.sim_paths, self.init_struc_paths):
            _trajs.append(Trajectory(_path))
            _mixes.append(read(_mix))
        self.trajs = _trajs
        self.init_strucs = _mixes

    @staticmethod
    def _count_H_and_O(atoms, n_iro2=192, n_h2o=400): # Counts extra oxygens and hydrogens on surface
        ans            = atoms.get_atomic_numbers()
        unique, counts = np.unique(ans, return_counts=True)
        dict_z_count   = dict(zip(unique,counts))
        n_ir = dict_z_count[77] - n_iro2
        n_o  = dict_z_count[8]  - 2*n_iro2 - n_h2o
        n_h  = dict_z_count[1]  - 2*n_h2o
        return {"Ir": n_ir, "O": n_o, "H": n_h}

    def calc_free_energy(self, atoms): # Balancing to get interface free energy
        # TO DO: deal with OH aqeous AND solvation/stablization
        G_h2o  = -13.907 #eV
        G_h2   = -6.745  #eV
        dict_counts = self._count_H_and_O(atoms)
        x_H2O = dict_counts['O']
        y_H2  = (2*x_H2O - dict_counts['H'])/2
        return atoms.info['md_energy'] + y_H2*G_h2 - x_H2O*G_h2o

    def _identify_cus_bri(self, idx):
        atoms = self.init_strucs[idx]
        max_Ir_Z = max([atom.z for atom in atoms if atom.symbol == 'Ir'])
        #These Iridiums are expected to stay put
        Ir_bri = np.array([atom.index for atom in atoms if (atom.symbol == 'Ir') and (atom.z == max_Ir_Z)])
        Ir_cus = np.array([atom.index for atom in atoms if (atom.symbol == 'Ir' and (np.round(max_Ir_Z - atom.z, 4) == 0.0953) and atom.index not in Ir_bri)])
        #Everything else can be dynamic and is counted in a general way
        O_cus = np.array([atom.index for atom in atoms if (atom.symbol == 'O' and atom.z > max_Ir_Z and min(atoms.get_distances(atom.index, Ir_cus, mic=True))<2.4 and min(atoms.get_distances(atom.index, Ir_bri, mic=True))>2.4)])
        return max_Ir_Z, Ir_bri, Ir_cus, O_cus

    def _count_bridge(self, atoms, Ir_bri, Ir_cus, max_Ir_Z):
        O_bri = np.array([atom.index for atom in atoms if (atom.symbol == 'O' and atom.z > max_Ir_Z and min(atoms.get_distances(atom.index, Ir_bri, mic=True))<2.4 and min(atoms.get_distances(atom.index, Ir_cus, mic=True))>2.4)])
        H_interface = np.array([atom.index for atom in atoms if (atom.symbol == 'H') and (atom.z < max_Ir_Z+4.0) and (atom.z > max_Ir_Z)])
        H_bri = np.array([atom.index for atom in atoms if (atom.index in H_interface ) and min(atoms.get_distances(atom.index, O_bri, mic=True))<1.1])
        bri_O_H2, bri_O_H, bri_O_c =[], [], []
        if len(H_bri)>0:
            bri_O_H2 = [atom.index for atom in atoms if atom.index in O_bri and len([d for d in atoms.get_distances(atom.index, H_bri, mic=True) if d<1.1])==2]
            bri_O_H = [atom.index for atom in atoms if atom.index in O_bri and len([d for d in atoms.get_distances(atom.index, H_bri, mic=True) if d<1.1])==1]
            bri_O_c = [atom.index for atom in atoms if atom.index in O_bri and len([d for d in atoms.get_distances(atom.index, H_bri, mic=True) if d<1.1])==0]
        else:
            bri_O_c = O_bri
        return atoms.info['time'], atoms.info['md_energy'], O_bri, bri_O_H2, bri_O_H, bri_O_c

    def _count_cus(self, atoms, Ir_bri, Ir_cus, max_Ir_Z):
        O_cus = np.array([atom.index for atom in atoms if (atom.symbol == 'O' and atom.z > max_Ir_Z and min(atoms.get_distances(atom.index, Ir_cus, mic=True))<2.4 and min(atoms.get_distances(atom.index, Ir_bri, mic=True))>2.4)])
        H_interface = np.array([atom.index for atom in atoms if (atom.symbol == 'H') and (atom.z < max_Ir_Z+4.0) and (atom.z > max_Ir_Z)])
        O_interface = np.array([atom.index for atom in atoms if (atom.symbol == 'O') and (atom.z < max_Ir_Z+4.0) and (atom.z > max_Ir_Z)])

        cus_O_H2, cus_O_H2, cus_O_c, cus_O_O =[], [], [], []
        cus_O_H2 = [atom.index for atom in atoms if atom.index in O_cus and len([d for d in atoms.get_distances(atom.index, H_interface, mic=True) if d<1.1])==2]
        cus_O_H  = [atom.index for atom in atoms if atom.index in O_cus and len([d for d in atoms.get_distances(atom.index, H_interface, mic=True) if d<1.1])==1]
        cus_O_c  = [atom.index for atom in atoms if atom.index in O_cus and len([d for d in atoms.get_distances(atom.index, H_interface, mic=True) if d<1.1])==0 and len([d for d in atoms.get_distances(atom.index, O_interface, mic=True) if d<1.7 and d>0.0])==0]
        cus_O_O  = [atom.index for atom in atoms if atom.index in O_cus and len([d for d in atoms.get_distances(atom.index, H_interface, mic=True) if d<1.1])==0 and len([d for d in atoms.get_distances(atom.index, O_interface, mic=True) if d<1.7 and d>0.0])==1]
        return atoms.info['time'], atoms.info['md_energy'], O_cus, cus_O_H2, cus_O_H, cus_O_c, cus_O_O

    def build_surf_compositions(self, path_save_csv=None,
                                make_plots_bars=False,
                                make_plots_lines=True,
                                path_save_plot=None):
        for i, index in enumerate(range(self.sim_indices[0], 1+self.sim_indices[1], 1)):
            print("Building surface comp df for index ", index)
            _max_Ir_Z, _Ir_bri, _Ir_cus, _O_cus = self._identify_cus_bri(idx=i)
            bri_df, cus_df = [], []
            for atoms in self.trajs[i][::self.skip_n_frames]:
                free_ener = self.calc_free_energy(atoms)
                time, ener, O_bri, bri_O_H2, bri_O_H, bri_O_c =  self._count_bridge(atoms=atoms, Ir_bri=_Ir_bri, Ir_cus=_Ir_cus, max_Ir_Z=_max_Ir_Z)
                append_this_bri = pd.DataFrame([{'Time': time, 'Tot_Energy': ener, 'Free_Energy':free_ener, 'OH2': len(bri_O_H2), 'OH': len(bri_O_H), 'O': len(bri_O_c), 'All': len(O_bri)}])
                bri_df.append(append_this_bri)
                time, ener, O_cus, cus_O_H2, cus_O_H, cus_O_c, cus_O_O =  self._count_cus(atoms=atoms, Ir_bri=_Ir_bri, Ir_cus=_Ir_cus, max_Ir_Z=_max_Ir_Z)
                append_this_cus = pd.DataFrame([{'Time': time, 'Tot_Energy': ener, 'Free_Energy':free_ener, 'OH2': len(cus_O_H2), 'OH': len(cus_O_H), 'O': len(cus_O_c), 'OO': len(cus_O_O), 'All': len(O_cus)}])
                cus_df.append(append_this_cus)
            bri_df = pd.concat(bri_df)
            cus_df = pd.concat(cus_df)
            filepath = Path.cwd()/'CSVs'
            filepath.mkdir(parents=True, exist_ok=True)
            bri_df.to_csv(filepath/f'bri_df_{index}.csv', sep= '\t')
            cus_df.to_csv(filepath/f'cus_df_{index}.csv', sep= '\t')
            if make_plots_bars or make_plots_lines:
                self.plot_surf_comps(bri_df=bri_df, cus_df=cus_df, sim_index=index,
                                     make_plots_bars=make_plots_bars,
                                     make_plots_lines=make_plots_lines,
                                     path_save_plot=path_save_plot)

    def _count_water(self, atoms, Ir_bri, Ir_cus, max_Ir_Z, start_height=0.0):
        H_water = np.array([atom.index for atom in atoms if (atom.symbol=='H' and atom.z>max_Ir_Z)])
        O_interface = np.array([atom.index for atom in atoms if (atom.symbol == 'O') and (atom.z < max_Ir_Z+4.0) and (atom.z > max_Ir_Z)])
        O_h2o = np.array([atom.index for atom in atoms if
                            (atom.symbol == 'O' and atom.z > max_Ir_Z+start_height and
                            min(atoms.get_distances(atom.index, Ir_bri, mic=True))>2.4 and
                            min(atoms.get_distances(atom.index, Ir_cus, mic=True))>2.4) and
                            len([d for d in atoms.get_distances(atom.index, H_water, mic=True) if d<1.1])==2 and
                            len([d for d in atoms.get_distances(atom.index, O_interface, mic=True) if d<1.7 and d>0.0])==0])
        O_oh  = np.array([atom.index for atom in atoms if
                            (atom.symbol == 'O' and atom.z > max_Ir_Z+start_height and
                            min(atoms.get_distances(atom.index, Ir_bri, mic=True))>2.4 and
                            min(atoms.get_distances(atom.index, Ir_cus, mic=True))>2.4) and
                            len([d for d in atoms.get_distances(atom.index, H_water, mic=True) if d<1.1])==1 and
                            len([d for d in atoms.get_distances(atom.index, O_interface, mic=True) if d<1.7 and d>0.0])==0])
        return O_h2o, O_oh

    def build_water_compositions(self, start_height=0.0, build_csv=False, make_plots=False):
        filepath = Path.cwd()/'CSVs'
        filepath.mkdir(parents=True, exist_ok=True)
        for i, index in enumerate(range(self.sim_indices[0], 1+self.sim_indices[1], 1)):
            if build_csv:
                print("Building water comp df for index ", index)
                _max_Ir_Z, _Ir_bri, _Ir_cus, _O_cus = self._identify_cus_bri(idx=i)
                water_df = []
                for atoms in self.trajs[i][::self.skip_n_frames]:
                    n_h2o, n_oh = self._count_water(atoms=atoms, Ir_bri=_Ir_bri,
                                                    Ir_cus=_Ir_cus, max_Ir_Z=_max_Ir_Z,
                                                    start_height=start_height)
                    append_this_bri = pd.DataFrame([{'Time': atoms.info['time'],
                                                     'Tot_Energy': atoms.info['md_energy'],
                                                     'Free_Energy':self.calc_free_energy(atoms),
                                                     'H2O': len(n_h2o),
                                                     'OH': len(n_oh),
                                                     'All': len(n_oh)+len(n_h2o)}])
                    water_df.append(append_this_bri)
                water_df = pd.concat(water_df)
                water_df.to_csv(filepath/f'water_df_{index}.csv', sep= '\t')
                if make_plots:
                    self.plot_water(water_df, sim_index=index)
            else:
                water_df = pd.read_csv(filepath/f'water_df_{index}.csv')
                if make_plots:
                    self.plot_water(water_df, sim_index=index)

    def plot_water(self, water_df, sim_index):
        plt.clf() #closes previous plots
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
        # fig.set_size_inches(6,5)
        ax1.plot(water_df['Time'], water_df['H2O'], label = '$H_2O$', color = 'orange', ls='-', alpha=1.0, zorder=0)
        ax2.plot(water_df['Time'], water_df['OH'],  label = '$OH^-$',  color = 'red',    ls='-', alpha=1.0, zorder=1)
        leg = fig.legend(loc='center', prop={'size': 15}, bbox_to_anchor=(0.5,0.5))
        leg._legend_box.align = "center"
        ax2.set(xlabel="$Time$ $in$ $ps$")
        fig.text(-0.01, 0.5, '$Count$', va='center', rotation='vertical')
        # # zoom-in / limit the view to different portions of the data
        ax2.set_ylim(-1, 5)
        ax1.set_ylim(395, 405)
        # hide the spines between ax and ax2
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.xaxis.tick_top()
        ax2.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()
        d = .015
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax2.plot((-d, +d), (-d, +d), zorder=3, **kwargs)        # top-left diagonal
        ax2.plot((1 - d, 1 + d), (-d, +d), zorder=4, **kwargs)  # top-right diagonal
        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        plt.plot((-d, +d), (1 - d, 1 + d), zorder=5, **kwargs)  # bottom-left diagonal
        plt.plot((1 - d, 1 + d), (1 - d, 1 + d), zorder=6, **kwargs)  # bottom-right diagonal

        filepath = Path.cwd()/'PLOTs'
        filepath.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath/f"water_comp_{sim_index}.png", dpi=300, bbox_inches='tight')

    def plot_surf_comps(self, bri_df, cus_df, path_save_plot, sim_index, make_plots_bars, make_plots_lines):
        bar_width = 5.0
        fig = plt.figure()
        fig.set_size_inches(12,5)
        gs = fig.add_gridspec(1, 2, hspace=0, wspace=0.1)
        (ax1, ax2) = gs.subplots(sharex='col', sharey='row')

        filepath = Path.cwd()/'PLOTs' if path_save_plot == None else path_save_plot
        filepath.mkdir(parents=True, exist_ok=True)
        ax1.set_title(f"$Bridge-site$")
        ax2.set_title(f"$Cus-site$")
        if make_plots_bars:
            ax1.bar(bri_df['Time'], bri_df['OH'],  width=bar_width, label = '$OH$', bottom=bri_df['OH2'], color = 'orange')
            ax1.bar(bri_df['Time'], bri_df['O'],   width=bar_width, label = '$O$', bottom=bri_df['OH2'] + bri_df['OH'], color = 'red')

            ax2.bar(cus_df['Time'], cus_df['OH2'], width=bar_width, label = '$OH_{2}$', color = 'green')
            ax2.bar(cus_df['Time'], cus_df['OH'],  width=bar_width, label = '$OH$', bottom=cus_df['OH2'], color = 'orange')
            ax2.bar(cus_df['Time'], cus_df['O'],   width=bar_width, label = '$O$',  bottom=cus_df['OH2'] + cus_df['OH'], color = 'red')
            ax2.bar(cus_df['Time'], cus_df['OO'],  width=bar_width, label = '$OOH$', bottom=cus_df['OH2'] + cus_df['OH'] + cus_df['O'], color = 'blue')

        if make_plots_lines:
            ax1.plot(bri_df['Time'], bri_df['OH'], label = '$OH$',      color = 'orange', ls='-', alpha=1.0)
            ax1.plot(bri_df['Time'], bri_df['O'],  label = '$O$',       color = 'red',    ls='-', alpha=0.5)

            ax2.plot(cus_df['Time'], cus_df['OH2'],label = '$OH_{2}$', color = 'green',  ls='-', alpha=0.5)
            ax2.plot(cus_df['Time'], cus_df['OH'], label = '$OH$',     color = 'orange', ls='-', alpha=1.0)
            ax2.plot(cus_df['Time'], cus_df['O'],  label = '$O$',      color = 'red',    ls='-', alpha=0.5)
            ax2.plot(cus_df['Time'], cus_df['OO'], label = '$OO$',     color = 'blue',   ls="-", alpha=0.5)

        for ax in [ax1,ax2]:
            ax.set_yticks(range(0,25,2))
            ax.grid(axis='y')
            ax.set(xlabel='Time in ps', ylabel=f"Coverage (out of 24 sites)")
            ax.label_outer()
        leg = plt.legend(loc='lower center',prop={'size': 18}, bbox_to_anchor=(-0.05,-0.35), ncol=4)
        leg._legend_box.align = "left"
        plt.savefig(filepath/f"surf_comps_{sim_index}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_bars(self, bri_df, cus_df, sim_index):
        bar_width = 5.0
        fig = plt.figure()
        fig.set_size_inches(12,5)
        gs = fig.add_gridspec(1, 2, hspace=0, wspace=0.1)
        (ax1, ax2) = gs.subplots(sharex='col', sharey='row')

        ax1.bar(bri_df['Time'], bri_df['OH'],  width=bar_width, label = '$OH$', bottom=bri_df['OH2'], color = 'orange')
        ax1.bar(bri_df['Time'], bri_df['O'],   width=bar_width, label = '$O$', bottom=bri_df['OH2'] + bri_df['OH'], color = 'red')
        ax1.set_title(f"$Bridge-site$")
        ax2.bar(cus_df['Time'], cus_df['OH2'], width=bar_width, label = '$OH_{2}$', color = 'green')
        ax2.bar(cus_df['Time'], cus_df['OH'],  width=bar_width, label = '$OH$', bottom=cus_df['OH2'], color = 'orange')
        ax2.bar(cus_df['Time'], cus_df['O'],   width=bar_width, label = '$O$',  bottom=cus_df['OH2'] + cus_df['OH'], color = 'red')
        ax2.bar(cus_df['Time'], cus_df['OO'],  width=bar_width, label = '$OOH$', bottom=cus_df['OH2'] + cus_df['OH'] + cus_df['O'], color = 'blue')
        ax2.set_title(f"$Cus-site$")

        for ax in [ax1,ax2]:
            ax.set_yticks(range(0,25,4))
            ax.grid(axis='y')
            ax.set(xlabel='Time in ps', ylabel=f"Coverage (out of 24 sites)")
            ax.label_outer()
        leg = plt.legend(loc='lower center',prop={'size': 18}, bbox_to_anchor=(-0.05,-0.35), ncol=4)
        leg._legend_box.align = "left"
        filepath = Path.cwd()/'PLOTs'
        filepath.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath/f"surf_comps_{sim_index}.png", dpi=300, bbox_inches='tight')

    def plot_density_profile(self):
        print("Plotting density profile")
        traj_all = []
        for traj in self.trajs:
            for im in traj[100:]:
                traj_all.append(im)
        z_O, z_H = [], []
        for im in traj_all:
            ind_O = [atom.index for atom in im if (atom.symbol=='O' and
                                                   0.5+self.iridium_surf_Zs[0]<atom.z<self.iridium_surf_Zs[1]-0.5)]
            z_O.append(np.array(im.get_positions()[ind_O, 2]))
            ind_H = [atom.index for atom in im if (atom.symbol=='H' and
                                                   0.5+self.iridium_surf_Zs[0]<atom.z<self.iridium_surf_Zs[1]-0.5)]
            z_H.append(np.array(im.get_positions()[ind_H, 2]))
        z_O, z_H = np.hstack(z_O), np.hstack(z_H)
        z_O = [z-self.iridium_surf_Zs[0] for z in z_O]
        z_H = [z-self.iridium_surf_Zs[0] for z in z_H]
        # For conversion of number to mass density
        from scipy.constants import N_A
        Norm_H2O = (1./18.)*N_A/(10**(24))
        Z_box = self.init_strucs[0].get_cell()[2,2]
        Vbin = (self.init_strucs[0].get_volume()/Z_box)*0.05
        z_O, z_H = np.asarray(z_O), np.asarray(z_H)
        hist_O, bins_O = np.histogram(z_O, range=[0, Z_box],
                                      bins=int(Z_box/0.05))
        hist_H, bins_H = np.histogram(z_H, range=[0, Z_box],
                                      bins=int(Z_box/0.05))
        hist_O, hist_H = np.insert(hist_O, 0, 0), np.insert(hist_H, 0, 0)
#        rho_O = hist_O/len(traj_all)/Vbin/Norm_H2O #Output as g/ml
#        rho_H = hist_H/len(traj_all)/Vbin/Norm_H2O #Output as g/ml
        rho_O = hist_O/len(traj_all)/Vbin #Output as number_count
        rho_H = hist_H/len(traj_all)/Vbin #Output as number_count
        from matplotlib.ticker import MultipleLocator
        plt.plot(bins_O, rho_O, lw=0.5, color='k', label='O')
        plt.plot(bins_H, rho_H, lw=0.5, color='gray', label='H')
        plt.xlabel('Z-distance in Ang.')
        plt.ylabel('Number Density')
        plt.xlim(0, self.iridium_surf_Zs[1]-self.iridium_surf_Zs[0])
        plt.xticks(np.linspace(0, self.iridium_surf_Zs[1]-self.iridium_surf_Zs[0], 5))
        plt.legend()
#        plt.yticks(range(0, int(np.ceil(max(rho_O))),1))
        filepath = Path.cwd()/'TRENDs'
        filepath.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath/f"density.png", dpi=300, bbox_inches='tight')
# %%
def main():
    sim_indices = int(input("Provide initial index:\n")), int(input("Provide final index:\n"))
    #sim_indices = 18, 19
    sim_paths, init_struc_paths = [], []
    for ind in range(sim_indices[0], 1+sim_indices[1], 1):
        sim_paths.append(f"sim_{ind}/concat.traj")
        init_struc_paths.append(f"sim_{ind}/mix.xyz")
    simulationGroup = TrajProcessor(sim_indices=sim_indices,
                                    sim_paths=sim_paths,
                                    init_struc_paths=init_struc_paths,
                                    iridium_surf_Zs = [4.39915211, 41.15683543],
                                    skip_n_frames=200)
#    simulationGroup.build_surf_compositions(path_save_csv=Path.cwd()/'CSVs',
#                                            make_plots_bars=False, make_plots_lines=True,
#                                            path_save_plot=Path.cwd()/'PLOTs')
#    simulationGroup.build_water_compositions(build_csv=True, make_plots=True, start_height=6.0)
    simulationGroup.plot_density_profile()

if __name__ == "__main__":
    main()
