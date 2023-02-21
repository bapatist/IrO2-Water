#%% 
from dataclasses import dataclass
from pathlib import Path
import glob
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
    sim_paths: list
    init_struc_paths: list
    skip_n_frames: int
    def __post_init__(self) -> None:
        """Reads all trajectories and init_strucs"""
        from ase.io import read
        _trajs, _mixes = [], []
        for _path, _mix in zip(self.sim_paths, self.init_struc_paths):
            _trajs.append(read(_path, ":"))
            _mixes.append(read(_mix))
        self.trajs = _trajs
        self.n_trajs = len(_trajs)
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
        O_cus = np.array([atom.index for atom in atoms if (atom.symbol == 'O' and atom.z > max_Ir_Z and min(atoms.get_distances(atom.index, Ir_cus, mic=True))<2.3 and min(atoms.get_distances(atom.index, Ir_bri, mic=True))>2.3)])
        return max_Ir_Z, Ir_bri, Ir_cus, O_cus
    
    def _count_bridge(self, atoms, Ir_bri, Ir_cus, max_Ir_Z):
        O_bri = np.array([atom.index for atom in atoms if (atom.symbol == 'O' and atom.z > max_Ir_Z and min(atoms.get_distances(atom.index, Ir_bri, mic=True))<2.3 and min(atoms.get_distances(atom.index, Ir_cus, mic=True))>2.3)])
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
    
    def build_compositions_dfs(self, make_plots=True):
        for index in range(self.n_trajs):
            print("Building comp df for index ", index)
            _max_Ir_Z, _Ir_bri, _Ir_cus, _O_cus = self._identify_cus_bri(idx=index)
            bri_df, cus_df = [], []
            for atoms in self.trajs[index][::self.skip_n_frames]:
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
            bri_df.to_csv(filepath/f'bri_df_{index}.csv')
            cus_df.to_csv(filepath/f'cus_df_{index}.csv')
            if make_plots:
                self.plot_bars(bri_df=bri_df, cus_df=cus_df, sim_index=index)

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
            ax.set(xlabel='Time in ps', ylabel=f"Coverage{chr(37)}")
            ax.label_outer()
        leg = plt.legend(loc='lower center',prop={'size': 18}, bbox_to_anchor=(-0.05,-0.35), ncol=4)
        leg._legend_box.align = "left"
        filepath = Path.cwd()/'PLOTs'  
        filepath.mkdir(parents=True, exist_ok=True)  
        plt.savefig(filepath/f"comps_{sim_index}.png", dpi=300, bbox_inches='tight')
# %%
def main():
    simulationGroup = TrajProcessor(sim_paths=glob.glob("sims/sim_*/newmd.xyz"),
                                    init_struc_paths=glob.glob("sims/sim_*/mix.xyz"),
                                    skip_n_frames=10)
    simulationGroup.build_compositions_dfs(make_plots=False)
# %%
if __name__ == "__main__":
    main()
# %%