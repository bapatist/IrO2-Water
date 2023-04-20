# %%
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
font = {'size'   : 18}
#matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
# %%
@dataclass
class time_corr():
    sim_indices: list # two numbers: init and final index
    sim_paths: list
    init_struc_paths: list
    skip_init_time: float
    skip_traj_frames: int
    def __post_init__(self) -> None:
        """Reads all trajectories and init_strucs"""
        from ase.io.trajectory import Trajectory
        from ase.io import read
        _trajs, _mixes = [], []
        for _path, _mix in zip(self.sim_paths, self.init_struc_paths):
            _trajs.append(Trajectory(_path)[::self.skip_traj_frames])
            _mixes.append(read(_mix))
        self.init_strucs = _mixes
        self.dt = np.round(_trajs[0][1].info['time'] - _trajs[0][0].info['time'], 2)
        skip_init_frames = int(self.skip_init_time/self.dt)
        self.trajs = [traj[skip_init_frames:] for traj in _trajs]
        print("dt = ", self.dt, "ps")
    
    def _id_first_layer(self, reac_state, struc):
        max_Ir_Z = max([atom.z for atom in struc if atom.symbol == 'Ir'])
        O_first = np.array([atom.index for atom in struc if (atom.symbol == 'O') and
                            (max_Ir_Z+ reac_state[0]) < atom.z < max_Ir_Z+reac_state[1]]) 
        return max_Ir_Z, O_first

    def _calc_time_corr(self, traj, time_delay, reac_state, prod_state):
        frame_delay = int(time_delay/self.dt)
        if frame_delay==0:
            traj_for_use = traj
        else:
            traj_for_use = traj[:-frame_delay]
        time_sum_of_num_avg = 0
        for idx, im in enumerate(traj_for_use):
            max_Ir_Z, id_oxys = self._id_first_layer(struc=im, reac_state=reac_state)
            p_reac, p_prod = 1, 0
            p_reacXprod, counter = 0, 0
            for id in id_oxys:
                if isinstance(prod_state, (int, float)):
                    if (traj[idx+frame_delay][id].z > max_Ir_Z+prod_state):
                        p_prod = 1
                    else:
                        p_prod = 0
                if prod_state==None:
                    if (max_Ir_Z+reac_state[0] < traj[idx+frame_delay][id].z < max_Ir_Z+reac_state[1]):
                        p_prod = 0
                    else:
                        p_prod = 1
                if isinstance(prod_state, list):
                    if (max_Ir_Z+prod_state[0] < traj[idx+frame_delay][id].z < max_Ir_Z+prod_state[1]):
                        p_prod = 0
                    else:
                        p_prod = 1
                counter += 1
                p_reacXprod += p_reac*p_prod    
            num_avg = p_reacXprod/counter
            time_sum_of_num_avg += num_avg
        time_avg = time_sum_of_num_avg/len(traj_for_use)
        return 1 - time_avg

    def plot_time_corr(self, time_lim, reac_states, prod_states, labels):
        x_axis = np.arange(0, time_lim, self.dt)
        fig, ax = plt.subplots()
        for reac_state, prod_state, label in zip(reac_states, prod_states, labels):
            Y_set = np.zeros(shape=(len(self.trajs), len(x_axis)))
            for i, traj in enumerate(self.trajs):
                print(f"Calculating for: reac_state, {reac_state}, trajectory {i}")
                Y_set[i] = [self._calc_time_corr(traj = traj, time_delay=x, 
                                                 reac_state=reac_state, prod_state=prod_state) 
                                                 for x in x_axis]
            y_axis = np.mean(Y_set, axis=0)
            ax.plot(x_axis, y_axis, marker='.', label=label)
            ax.set(xlabel="t, delay in ps", ylabel="C(t)")
            ax.legend()
            # ax.set_ylim(0.6,1.01)
            plt.savefig("tc.png", dpi=300, bbox_inches='tight')

# %%
def main():
    sim_indices = [0,7]
    sim_paths, init_struc_paths = [], []
    for ind in range(sim_indices[0], 1+sim_indices[1], 1):
        sim_paths.append(f"./sim_{ind}/newmd.traj")
        init_struc_paths.append(f"./sim_{ind}/mix.xyz")

    TC = time_corr(sim_indices=sim_indices,
                   sim_paths=sim_paths,
                   init_struc_paths=init_struc_paths,
                   skip_init_time=10.0,
                   skip_traj_frames=100 # original dt = 0.1 ps, 10 frames=1ps
                   )
    TC.plot_time_corr(time_lim = 200.0, 
                      reac_states=[[2.6, 3.75], [5.75, 6.85], [15.0, 16.15]],
                      prod_states=[3.75, [4.75, 7.85], [14.0, 17.15]], # one number means above that is product, list of two means outside this range is product
                      labels = ["Ist layer", "IInd layer", "Bulk"]
                      )

if __name__ == "__main__":
    main()
# %%