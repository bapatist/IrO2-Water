# %%
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
class time_corr():
    first_layer_lims: list # init and final distance from surface
    sim_indices: list # two numbers: init and final index
    sim_paths: list
    init_struc_paths: list
    skip_init_time: float
    def __post_init__(self) -> None:
        """Reads all trajectories and init_strucs"""
        from ase.io.trajectory import Trajectory
        from ase.io import read
        _trajs, _mixes = [], []
        for _path, _mix in zip(self.sim_paths, self.init_struc_paths):
            _trajs.append(Trajectory(_path))
            _mixes.append(read(_mix))
        self.init_strucs = _mixes
        self.dt = np.round(_trajs[0][1].info['time'] - _trajs[0][0].info['time'], 2)
        skip_init_frames = int(self.skip_init_time/self.dt)
        self.trajs = [traj[skip_init_frames:] for traj in _trajs]

    
    def _id_first_layer(self, struc):
        max_Ir_Z = max([atom.z for atom in struc if atom.symbol == 'Ir'])
        O_first = np.array([atom.index for atom in struc if (atom.symbol == 'O') and
                            (max_Ir_Z+self.first_layer_lims[0]) < atom.z < max_Ir_Z+self.first_layer_lims[1]]) 
        return max_Ir_Z, O_first

    def _calc_time_corr(self, time_delay):
        # lets try first for only one trajectory
        frame_delay = int(time_delay/self.dt)
        traj = self.trajs[0]
        max_Ir_Z, id_oxys = self._id_first_layer(struc=traj[0])
        molecule_sum = 0
        for id in id_oxys:
            p_reac, p_prod = 0, 0
            p_reacXprod, counter = 0, 0
            for ind, im in enumerate(traj):
                try:
                    if (max_Ir_Z+self.first_layer_lims[0] < im[id].z < max_Ir_Z+self.first_layer_lims[1]):
                        p_reac = 1
                    else:
                        p_reac = 0
                    if (max_Ir_Z+self.first_layer_lims[0] < traj[ind+frame_delay][id].z < max_Ir_Z+self.first_layer_lims[1]):
                        p_prod = 0
                    else:
                        p_prod = 1
                    counter += 1
                except IndexError:
                    break
                p_reacXprod += p_reac*p_prod
            #print(len(traj), frame_delay)
            time_avg = p_reacXprod/counter
            # time_avg = p_reacXprod/(len(traj)-frame_delay) # divide by N-n (# of pairs of t=0, t=t)
            molecule_sum += time_avg
        molecule_avg = molecule_sum/len(id_oxys)
        return (1 - molecule_avg)

    def plot_time_corr(self, time_lim=1):
        x_axis = np.arange(0, time_lim, self.dt)
        y_axis = [self._calc_time_corr(time_delay=x) for x in x_axis]
        fig, ax = plt.subplots()
        ax.plot(x_axis, y_axis, marker='.')
        ax.set(xlabel="$Time$ $delay$", ylabel="$C(t)$")
        ax.set_ylim(0.90,1.01)
        plt.savefig("tc.png", dpi=300, bbox_inches='tight')

# %%
def main():
    first_layer_lims = [2.6, 4.9]
    sim_indices = [1,7]
    sim_paths, init_struc_paths = [], []
    for ind in range(sim_indices[0], 1+sim_indices[1], 1):
        sim_paths.append(f"/work/surf/residence_lt/sim_{ind}/newmd.traj")
        init_struc_paths.append(f"/work/surf/residence_lt/sim_{ind}/mix.xyz")

    TC = time_corr(first_layer_lims=first_layer_lims,
                   sim_indices=sim_indices,
                   sim_paths=sim_paths,
                   init_struc_paths=init_struc_paths,
                   skip_init_time=10.0
                   )
    TC.plot_time_corr(time_lim = 25.0)

if __name__ == "__main__":
    main()
# %%
