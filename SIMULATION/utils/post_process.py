#%%
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
font = {'size'   : 18}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
# %%
@dataclass
class TrajProcessor():
    sim_indices: list # two numbers: init and final index
    sim_paths: list
    init_struc_paths: list

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
        # Getting surface iridum's Z values, ASSUMING ALL MIX.xyz HAVE SAME SURFACE POSITIONS
        ref_mix = _mixes[0]
        bot_ir_zs = [atom.z for atom in ref_mix if atom.symbol=='Ir' and atom.z<(ref_mix.get_cell()[2,2])/2]
        top_ir_zs = [atom.z for atom in ref_mix if atom.symbol=='Ir' and atom.z>(ref_mix.get_cell()[2,2])/2]
        self.iridium_surf_Zs = [max(bot_ir_zs), min(top_ir_zs)]

    @staticmethod
    def _count_H_and_O(atoms, n_iro2=240, n_h2o=420): # Counts extra oxygens and hydrogens on surface
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
# Plot energy tracking
    def build_energy_tracker(self, skip_n_frames, run_avg_window, build_csv, make_plots):
        def _plot_energy_tracker(df, window_size, window_in_ps, sim_index):
            df['Running_Pot_Energy'] = df['Pot_Energy'].rolling(window=window_size).mean()
            fig, ax = plt.subplots()

            ax.plot(df['Time'], df['Pot_Energy'], label='Original Data', lw=0.2)
            ax.plot(df['Time'], df['Running_Pot_Energy'], label=f"Running avg ({window_in_ps} ps)")
            ax.set(xlabel='$\mathrm {Time,\ ps}$',ylabel='$\mathrm {Pot.\ energy,\ eV}$')
            ax.legend()
            plt.savefig(f"PLOTs/energy_{sim_index}.png", dpi=300, bbox_inches='tight')
        if build_csv:
            print(f"Building csv for energy tracking. Skipping every {skip_n_frames} frames")
            for i, index in enumerate(range(self.sim_indices[0], 1+self.sim_indices[1], 1)):
                print(f"Building csv for index, {index}")
                df = []
                for atoms in self.trajs[i][::skip_n_frames]:
                    time        = format(atoms.info['time'], '.2f')
                    tot_energy  = format(atoms.info['md_energy'], '4f')
                    pot_energy  = format(atoms.info['pot_energy'], '4f')
                    free_energy = format(self.calc_free_energy(atoms), '4f')
                    df.append(pd.DataFrame([{'Time': time,
                                             'Tot_Energy': tot_energy,
                                             'Pot_Energy': pot_energy,
                                             'Free_Energy':free_energy}]))
                df = pd.concat(df, ignore_index=True)
                df = df.apply(pd.to_numeric)
                filepath = Path.cwd()/'CSVs'
                filepath.mkdir(parents=True, exist_ok=True)
                df.to_csv(filepath/f'energy_{index}.csv', sep= '\t', index=True)
                print(f"Finished writing CSV to {filepath/f'energy_{index}.csv'}")
                if make_plots:
                    timestep     = df['Time'][1] - df['Time'][0]
                    window_in_ps = format(run_avg_window*timestep, '.1f')
                    _plot_energy_tracker(df, window_size=run_avg_window, window_in_ps=window_in_ps, sim_index=index)


# Surface composition functions
    def _identify_cus_bri(self, idx):
        atoms = self.init_strucs[idx]
        ids_O_cus, ids_O_bri = [], []
        # O_cus_Z = [np.round(self.iridium_surf_Zs[0]+1.99074, 2),
        #            np.round(self.iridium_surf_Zs[1]-2.00916, 2)]
        O_cus_Z = [np.round(self.iridium_surf_Zs[0]+2.0),
                   np.round(self.iridium_surf_Zs[1]-2.0)]
        O_bri_Z = [np.round(self.iridium_surf_Zs[0]+1.23, 2),
                   np.round(self.iridium_surf_Zs[1]-1.23, 2)]
        for atom in atoms:
            if atom.symbol == 'O':
                if np.round(atom.z) in O_cus_Z:
                    ids_O_cus.append(atom.index)
                elif np.round(atom.z, 2) in O_bri_Z:
                    ids_O_bri.append(atom.index)
        return ids_O_cus, ids_O_bri

    def OLD_count_bridge(self, atoms, O_bri):
        H_interface = np.array([atom.index for atom in atoms if
                                (atom.symbol == 'H') and
                                ((atom.z < self.iridium_surf_Zs[0]+4.0) or
                                 (atom.z > self.iridium_surf_Zs[1]-4.0))])
        H_bri = np.array([atom.index for atom in atoms if
                            atom.index in H_interface and
                            min(atoms.get_distances(atom.index, O_bri, mic=True))<=1.1])
        bri_OH2, bri_OH, bri_Oc = [], [], []
        if len(H_bri)>0:
            for atom in atoms:
                if atom.index in O_bri:
                    no_of_H = len([d for d in
                            atoms.get_distances(atom.index, H_bri, mic=True)
                            if d<1.1])
                    if no_of_H==2:
                        bri_OH2.append(atom.index)
                    if no_of_H==1:
                        bri_OH.append(atom.index)
                    if no_of_H==0:
                        bri_Oc.append(atom.index)
        else:
            bri_Oc = O_bri
        return atoms.info['time'], atoms.info['md_energy'], bri_OH2, bri_OH, bri_Oc

    def _count_cus_bot(self, atoms, ids_O_cus, surf_Zs, cell_z):
        # NOT YET APPLICABLE TO PEROXY, OOH SPECIES
        ids_H_nearby = np.array([atom.index for atom in atoms if (atom.symbol == 'H') and
                                 (surf_Zs[0] < atom.z < 4.0+surf_Zs[0])])
        n_o, n_oh, n_h2o, n_transit = 0, 0, 0, 0
        for id_O in ids_O_cus:
            if atoms[id_O].z < cell_z/2:
                oh_distances = np.array(atoms.get_distances(id_O, ids_H_nearby, mic=True))
                if (oh_distances<1.3).sum()==0:
                    n_o += 1
                elif (oh_distances<1.12).sum()==2:
                    n_h2o += 1
                elif (oh_distances<1.1).sum()==1 and (oh_distances<1.4).sum()==1:
                    n_oh += 1
                elif (oh_distances<1.1).sum()==3:
                    raise ValueError('Found an 3H < 1.1 angstrom on a cus site')
                else:
                    n_transit += 1
        return n_o, n_oh, n_h2o, n_transit

    def _count_cus_top(self, atoms, ids_O_cus, surf_Zs, cell_z):
        # NOT YET APPLICABLE TO PEROXY, OOH SPECIES
        ids_H_nearby = np.array([atom.index for atom in atoms if (atom.symbol == 'H') and
                                 (surf_Zs[1] > atom.z > surf_Zs[1]-4.0)])
        n_o, n_oh, n_h2o, n_transit = 0, 0, 0, 0
        for id_O in ids_O_cus:
            if atoms[id_O].z > cell_z/2:
                oh_distances = np.array(atoms.get_distances(id_O, ids_H_nearby, mic=True))
                if (oh_distances<1.3).sum()==0:
                    n_o += 1
                elif (oh_distances<1.12).sum()==2:
                    n_h2o += 1
                elif (oh_distances<1.1).sum()==1 and (oh_distances<1.4).sum()==1:
                    n_oh += 1
                elif (oh_distances<1.1).sum()==3:
                    raise ValueError('Found an 3H < 1.1 angstrom on a cus site')
                else:
                    n_transit += 1
        return n_o, n_oh, n_h2o, n_transit

    def build_surf_compositions(self, skip_n_frames, build_csv=False,
                                make_plots_bars=False,
                                make_plots_lines=True,
                                path_save_plot=None):
        for i, index in enumerate(range(self.sim_indices[0], 1+self.sim_indices[1], 1)):
            if build_csv:
                print("Building surface comp df for index ", index)
                O_cus_all, O_bri_all = self._identify_cus_bri(idx=i)
                print("Identified ", len(O_bri_all), "bridge and ", len(O_cus_all), "cus oxygens")
                ref_im = self.trajs[i][0]
                cell_z = ref_im.get_cell()[2,2]
                surf_Zs = [0,999]
                for atom in ref_im:
                    if atom.symbol == 'Ir':
                        if atom.z < cell_z/2:
                            surf_Zs[0] = max(surf_Zs[0], atom.z)
                        if atom.z > cell_z/2:
                            surf_Zs[1] = min(surf_Zs[1], atom.z)
                cus_df_bot, cus_df_top = [], []
                for atoms in self.trajs[i][::skip_n_frames]:
                    time        = format(atoms.info['time'], '.2f')
                    tot_energy  = format(atoms.info['md_energy'], '4f')
                    free_energy = format(self.calc_free_energy(atoms), '4f')
                    bot_n_o, bot_n_oh, bot_n_h2o, bot_n_transit = self._count_cus_bot(atoms=atoms,
                                                                                      ids_O_cus=O_cus_all,
                                                                                      surf_Zs=surf_Zs,
                                                                                      cell_z=cell_z)
                    top_n_o, top_n_oh, top_n_h2o, top_n_transit = self._count_cus_top(atoms=atoms,
                                                                                      ids_O_cus=O_cus_all,
                                                                                      surf_Zs=surf_Zs,
                                                                                      cell_z=cell_z)
                    cus_df_bot.append(pd.DataFrame([{'Time': time,
                                                     'Tot_Energy': tot_energy,
                                                     'Free_Energy':free_energy,
                                                     'O': bot_n_o,
                                                     'OH': bot_n_oh,
                                                     'H2O': bot_n_h2o,
                                                     'O_transit': bot_n_transit,
                                                     'All': bot_n_o+bot_n_h2o+bot_n_oh+bot_n_transit}]))
                    cus_df_top.append(pd.DataFrame([{'Time': time,
                                                     'Tot_Energy': tot_energy,
                                                     'Free_Energy':free_energy,
                                                     'O': top_n_o,
                                                     'OH': top_n_oh,
                                                     'H2O': top_n_h2o,
                                                     'O_transit': top_n_transit,
                                                     'All': top_n_o+top_n_oh+top_n_h2o+top_n_transit}]))
                cus_df_bot = pd.concat(cus_df_bot)
                cus_df_top = pd.concat(cus_df_top)
                filepath = Path.cwd()/'CSVs'
                filepath.mkdir(parents=True, exist_ok=True)
                cus_df_bot.to_csv(filepath/f'cus_bot_{index}.csv', sep= '\t', index=False)
                cus_df_top.to_csv(filepath/f'cus_top_{index}.csv', sep= '\t', index=False)
                print(f"Finished writing CSVs to {filepath}")
                if make_plots_bars or make_plots_lines:
                    self._plot_surf_comps(cus_df_bot=cus_df_bot, cus_df_top=cus_df_top, sim_index=index,
                                        make_plots_bars=make_plots_bars,
                                        make_plots_lines=make_plots_lines,
                                        path_save_plot=path_save_plot)
            else:
                filepath = Path.cwd()/'CSVs'
                cus_df_bot = pd.read_csv(filepath/f'cus_bot_{index}.csv', sep= '\t')
                cus_df_top = pd.read_csv(filepath/f'cus_top_{index}.csv', sep= '\t')
                print(cus_df_bot.head())
                if make_plots_bars or make_plots_lines:
                    self._plot_surf_comps(cus_df_bot=cus_df_bot, cus_df_top=cus_df_top, sim_index=index,
                                        make_plots_bars=make_plots_bars,
                                        make_plots_lines=make_plots_lines,
                                        path_save_plot=path_save_plot)

    def _plot_surf_comps(self, cus_df_bot, cus_df_top, sim_index, make_plots_bars, make_plots_lines, path_save_plot):
        # Making time neater for plots
        cus_df_bot['Time'] = [int(float(t)) for t in cus_df_bot['Time']]
        cus_df_top['Time'] = [int(float(t)) for t in cus_df_top['Time']]
        bar_width = 5.0
        fig = plt.figure()
        fig.set_size_inches(12,5)
        gs = fig.add_gridspec(1, 2, hspace=0, wspace=0.1)
        (ax1, ax2) = gs.subplots(sharex='col', sharey='row')

        filepath = Path.cwd()/'PLOTs' if path_save_plot == None else path_save_plot
        filepath.mkdir(parents=True, exist_ok=True)
        ax1.set_title("$\mathrm {Lower\ surface}$")
        ax2.set_title("$\mathrm {Upper\ surface}$")
        if make_plots_bars:
            ax1.bar(cus_df_bot['Time'], cus_df_bot['H2O'], width=bar_width, label = '$\mathrm {H_2O*}$', color = 'green')
            ax1.bar(cus_df_bot['Time'], cus_df_bot['OH'],  width=bar_width, label = '$\mathrm {*OH}$', bottom=cus_df_bot['H2O'], color = 'orange')
            ax1.bar(cus_df_bot['Time'], cus_df_bot['O'],   width=bar_width, label = '$\mathrm {O*}$',  bottom=cus_df_bot['H2O'] + cus_df_bot['OH'], color = 'red')
            ax1.bar(cus_df_bot['Time'], cus_df_bot['O_transit'],  width=bar_width, label = '$\mathrm {*O_tH_x}$', bottom=cus_df_bot['H2O'] + cus_df_bot['OH'] + cus_df_bot['O'], color = 'blue')

            ax2.bar(cus_df_top['Time'], cus_df_top['H2O'], width=bar_width, label = '$\mathrm {H_2O*}$', color = 'green')
            ax2.bar(cus_df_top['Time'], cus_df_top['OH'],  width=bar_width, label = '$\mathrm {*OH}$', bottom=cus_df_top['H2O'], color = 'orange')
            ax2.bar(cus_df_top['Time'], cus_df_top['O'],   width=bar_width, label = '$\mathrm {O*}$',  bottom=cus_df_top['H2O'] + cus_df_top['OH'], color = 'red')
            ax2.bar(cus_df_top['Time'], cus_df_top['O_transit'],  width=bar_width, label = '$\mathrm {*O_tH_x}$', bottom=cus_df_top['H2O'] + cus_df_top['OH'] + cus_df_top['O'], color = 'blue')

        if make_plots_lines:
            ax1.plot(cus_df_bot['Time'], cus_df_bot['All'],label = '$\mathrm {*O_{all}}$',color = 'k',      ls="-", marker=".", alpha=0.5)
            ax1.plot(cus_df_bot['Time'], cus_df_bot['H2O'],label = '$\mathrm {H_2O*}$',   color = 'green',  ls='-', marker=".", alpha=0.5)
            ax1.plot(cus_df_bot['Time'], cus_df_bot['OH'], label = '$\mathrm {*OH}$',     color = 'orange', ls='-', marker=".", alpha=0.5)
            ax1.plot(cus_df_bot['Time'], cus_df_bot['O'],  label = '$\mathrm {O*}$',      color = 'red',    ls='-', marker=".", alpha=0.5)
            ax1.plot(cus_df_bot['Time'], cus_df_bot['O_transit'], label = '$\mathrm {*O_tH_x}$', color = 'blue',   ls="-", marker=".", alpha=0.5)

            ax2.plot(cus_df_top['Time'], cus_df_top['All'],label = '$\mathrm {*O_{all}}$',color = 'k',      ls="-", marker=".", alpha=0.5)
            ax2.plot(cus_df_top['Time'], cus_df_top['H2O'],label = '$\mathrm {H_2O*}$',   color = 'green',  ls='-', marker=".", alpha=0.5)
            ax2.plot(cus_df_top['Time'], cus_df_top['OH'], label = '$\mathrm {*OH}$',     color = 'orange', ls='-', marker=".", alpha=0.5)
            ax2.plot(cus_df_top['Time'], cus_df_top['O'],  label = '$\mathrm {O*}$',      color = 'red',    ls='-', marker=".", alpha=0.5)
            ax2.plot(cus_df_top['Time'], cus_df_top['O_transit'], label = '$\mathrm {*O_tH_x}$', color = 'blue',   ls="-", marker=".", alpha=0.5)

        for ax in [ax1,ax2]:
            ax.set_yticks(range(0,25,2))
            ax.grid(axis='y')
            ax.set(xlabel='$\mathrm {Time\ in\ ps}$', ylabel='$\mathrm {Coverage}$')
            ax.label_outer()
        leg = plt.legend(loc='lower center',prop={'size': 18}, bbox_to_anchor=(-0.05,-0.35), ncol=5)
        leg._legend_box.align = "left"
        plt.savefig(filepath/f"surf_comps_{sim_index}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

# Water composition functions
    def _count_water(self, atoms, ids_H, surf_Zs, d_from_surf):
        # atoms: ase.Atoms frame on which we will count H2Os and OHs
        # ids_H: list of ints, all hydrogen atom IDs in system
        # surf_Zs: list of two floats for surface-Zs (Z-bot and Z-top)
        n_h2o, n_oh, n_h3o, n_transit = 0, 0, 0, 0
        for atom in atoms:
            if atom.symbol == 'O' and (surf_Zs[0]+d_from_surf < atom.z < surf_Zs[1]-d_from_surf):
                oh_distances = np.array(atoms.get_distances(atom.index, ids_H, mic=True))
                if (oh_distances<1.12).sum()==2:
                    n_h2o += 1
                elif (oh_distances<1.1).sum()==1 and (oh_distances<1.4).sum()==1:
                    n_oh += 1
                elif (oh_distances<1.1).sum()==3:
                    n_h3o += 1
                else:
                    n_transit += 1
        return n_h2o, n_oh, n_h3o, n_transit

    def build_water_compositions(self, skip_n_frames, d_from_surf, build_csv=False, make_plots=False):
        filepath = Path.cwd()/'CSVs'
        filepath.mkdir(parents=True, exist_ok=True)
        ids_H = [atom.index for atom in self.init_strucs[0] if atom.symbol=='H']
        for i, index in enumerate(range(self.sim_indices[0], 1+self.sim_indices[1], 1)):
            if build_csv:
                print("Building water comp df for index ", index)
                ref_im = self.trajs[i][0]
                cell_z = ref_im.get_cell()[2,2]
                surf_Zs = [0,999]
                for atom in ref_im:
                    if atom.symbol == 'Ir':
                        if atom.z < cell_z/2:
                            surf_Zs[0] = max(surf_Zs[0], atom.z)
                        if atom.z > cell_z/2:
                            surf_Zs[1] = min(surf_Zs[1], atom.z)
                print(f"For sim_{i}, identified surfaces at Z={surf_Zs}")
                water_df = []
                for atoms in self.trajs[i][::skip_n_frames]:
                    n_h2o, n_oh, n_h3o, n_transit = self._count_water(atoms=atoms,
                                                    ids_H=ids_H,
                                                    surf_Zs=surf_Zs,
                                                    d_from_surf=d_from_surf)
                    append_this_bri = pd.DataFrame([{'Time': format(atoms.info['time'], '.2f'),
                                                     'Tot_Energy': format(atoms.info['md_energy'], '4f'),
                                                     'Free_Energy':format(self.calc_free_energy(atoms), '4f'),
                                                     'H2O': n_h2o,
                                                     'OH': n_oh,
                                                     'H3O': n_h3o,
                                                     'O_transit': n_transit,
                                                     'All': n_oh+n_h2o+n_transit}])
                    water_df.append(append_this_bri)
                water_df = pd.concat(water_df)
                water_df.to_csv(filepath/f'water_df_{index}.csv', sep= '\t', index=False)
                if make_plots:
                    self._plot_water(water_df, sim_index=index)
            else:
                water_df = pd.read_csv(filepath/f'water_df_{index}.csv')
                if make_plots:
                    self._plot_water(water_df, sim_index=index)

    def _plot_water(self, water_df, sim_index):
        plt.clf() #closes previous plots
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
        # fig.set_size_inches(6,5)
        ax1.plot([int(float(t)) for t in water_df['Time']], water_df['All'], label = '$\mathrm {all}$',    color = 'k',     ls='-', alpha=0.6, marker='.', zorder=0)
        ax1.plot([int(float(t)) for t in water_df['Time']], water_df['H2O'], label = '$\mathrm {H_2O}$',   color = 'green', ls='-', alpha=0.6, marker='.', zorder=0)
        ax2.plot([int(float(t)) for t in water_df['Time']], water_df['OH'],  label = '$\mathrm {OH^-}$',   color = 'orange',ls='-', alpha=0.6, marker='.', zorder=1)
        ax2.plot([int(float(t)) for t in water_df['Time']], water_df['H3O'], label = '$\mathrm {H_3O^+}$', color = 'aqua',  ls='-', alpha=0.6, marker='.', zorder=1)
        ax2.plot([int(float(t)) for t in water_df['Time']], water_df['O_transit'], label='$\mathrm {O_tH_x}$', color='green', ls='--', alpha=0.6, marker='.', lw=0.5, zorder=1)
        leg = fig.legend(loc='center', prop={'size': 15}, bbox_to_anchor=(0.5,0.5))
        leg._legend_box.align = "center"
        ax2.set(xlabel="$\mathrm {Time\ in\ ps}$")
        fig.text(-0.01, 0.5, '$\mathrm {Count}$', va='center', rotation='vertical')
        # # zoom-in / limit the view to different portions of the data
        #ax2.set_ylim(-1, 5)
        #ax1.set_ylim(300, 350)
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

    def plot_density_profile(self):
        print("\n Plotting density profile")
        print("\n WARNING: For Density Profile, try to provide only 1 simulation to avoid wrong surface-Z referencing")
        # Surface Z for the first traj
        ref_im = self.trajs[0][0]
        cell_z = ref_im.get_cell()[2,2]
        surf_Zs = [0,999]
        for atom in ref_im:
            if atom.symbol == 'Ir':
                if atom.z < cell_z/2:
                    surf_Zs[0] = max(surf_Zs[0], atom.z)
                if atom.z > cell_z/2:
                    surf_Zs[1] = min(surf_Zs[1], atom.z)
        print(f"Identified surfaces at Z={surf_Zs}")
        traj_all = []
        for traj in self.trajs:
            for im in traj[100:]:
                traj_all.append(im)
        z_O, z_H = [], []
        for im in traj_all:
            for atom in im:
                if atom.symbol == 'O' and 0.5+surf_Zs[0]<atom.z<surf_Zs[1]-0.5:
                    z_O.append(atom.position[-1])
                elif atom.symbol == 'H' and 0.5+surf_Zs[0]<atom.z<surf_Zs[1]-0.5:
                    z_H.append(atom.position[-1])
        z_O, z_H = np.hstack(z_O), np.hstack(z_H)

        z_O = [z-surf_Zs[0] for z in z_O]
        z_H = [z-surf_Zs[0] for z in z_H]
        # For conversion of number to mass density
        from scipy.constants import N_A
        Norm_H2O = (1./18.)*N_A/(10**(24))
        Z_box = ref_im.get_cell()[2,2]
        Vbin = (ref_im.get_volume()/Z_box)*0.05
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

        d_profile_arr = np.asarray([bins_O, rho_O, rho_H])
        filepath = Path.cwd()/'TRENDs'
        filepath.mkdir(parents=True, exist_ok=True)
        np.savetxt(filepath/"density_prof.csv", d_profile_arr, delimiter=',', fmt='%.3f')

        from matplotlib.ticker import MultipleLocator
        plt.plot(bins_O, rho_O, lw=1.0, color='r', label='O')
        plt.plot(bins_H, rho_H, lw=1.0, color='b', label='H')
        plt.xlabel('$\mathrm {Z-distance\ in\ \AA}$')
        plt.ylabel('$\mathrm {Number\ density}$')
        plt.xlim(0, surf_Zs[1]-surf_Zs[0])
        plt.xticks(np.linspace(0, surf_Zs[1]-surf_Zs[0], 5))
        plt.legend()
#        plt.yticks(range(0, int(np.ceil(max(rho_O))),1))
        plt.savefig(filepath/f"density.png", dpi=300, bbox_inches='tight')
# %%
def main():
    sim_indices = int(input("Provide initial index:\n")), int(input("Provide final index:\n"))
    # traj_file_name = input("Provide .traj file name to post process (example=concat.traj): ")
    #sim_indices = 18, 19
    sim_paths, init_struc_paths = [], []
    for ind in range(sim_indices[0], 1+sim_indices[1], 1):
        # sim_paths.append(f"sim_{ind}/{traj_file_name}")
        sim_paths.append(f"sim_{ind}/concat.traj")
        init_struc_paths.append(f"sim_{ind}/mix.xyz")
    simulationGroup = TrajProcessor(sim_indices=sim_indices,
                                    sim_paths=sim_paths,
                                    init_struc_paths=init_struc_paths)
    simulationGroup.build_surf_compositions(skip_n_frames=20,
                                            build_csv=True,
                                            make_plots_bars=False, make_plots_lines=True,
                                            path_save_plot=Path.cwd()/'PLOTs')
    simulationGroup.build_energy_tracker(skip_n_frames=10, run_avg_window=200,
                                         build_csv=True, make_plots=True)
    d_from_surf = float(input("Distance above surface to check water components(Reccommended=2.5):"))
    simulationGroup.build_water_compositions(skip_n_frames=20,
                                             d_from_surf=d_from_surf,
                                             build_csv=True,
                                             make_plots=True)
    simulationGroup.plot_density_profile()

if __name__ == "__main__":
    main()
