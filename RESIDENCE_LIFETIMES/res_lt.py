# %%
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
font = {'size'   : 18}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
# %%
@dataclass
class time_corr():
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
    
    def _id_react_layer(self, reac_state, struc):
        cell_mid_z     = (struc.cell[2][2])/2
        ir_bot_zs, ir_top_zs = [], []
        for z_ in [atom.z for atom in struc if atom.symbol == 'Ir']:
            if z_ < cell_mid_z:
                ir_bot_zs.append(z_)
            else:
                ir_top_zs.append(z_)
        iridium_extreme_zs = [np.max(ir_bot_zs), np.min(ir_top_zs)]

        id_oxys = []
        for atom in struc:
            if atom.symbol == 'O' and iridium_extreme_zs[0]<atom.z<iridium_extreme_zs[1]:
                closest_surf_dist=np.min(np.absolute(np.subtract(iridium_extreme_zs, atom.z)))
                if reac_state[0]<closest_surf_dist<reac_state[1]:
                    id_oxys.append(atom.index)

        return iridium_extreme_zs, id_oxys
    
    def _check_recross(self, traj_slice, id, ir_extreme_zs, prod_state):
        recrossed = False
        if isinstance(prod_state, list):
            for im in traj_slice:
                z_instance = im[id].z
                z_closest_surf = np.min(np.absolute(np.subtract(ir_extreme_zs, z_instance)))
                if (prod_state[0] > z_closest_surf or  
                    prod_state[1] < z_closest_surf):
                    recrossed = True
        return recrossed

    def _calc_time_corr(self, traj, time_delay, reac_state, prod_state):
        frame_delay = int(time_delay/self.dt)
        if frame_delay==0:
            traj_for_use = traj
        else:
            traj_for_use = traj[:-frame_delay]
        time_sum_of_num_avg = 0
        for idx, im in enumerate(traj_for_use):
            ir_extreme_zs, id_oxys=self._id_react_layer(struc=im, reac_state=reac_state)
            p_reac, p_prod = 1, 0
            p_reacXprod, counter = 0, 0
            for id in id_oxys:
                z_future = traj[idx+frame_delay][id].z
                z_closest_surf = np.min(np.absolute(np.subtract(ir_extreme_zs, z_future)))
                # if isinstance(prod_state, (int, float)):
                #     if z_closest_surf>prod_state:
                #         p_prod = 1
                #     elif reac_state[1]>z_closest_surf>prod_state:
                #         continue
                #     else:
                #         p_prod = 0
                # if prod_state==None:
                #     if (reac_state[0] < z_closest_surf < reac_state[1]):
                #         p_prod = 0
                #     else:
                #         p_prod = 1
                if isinstance(prod_state, list):
                    if (prod_state[0] > z_closest_surf or  
                        prod_state[1] < z_closest_surf):
                        p_prod = 1
                    elif (reac_state[0] < z_closest_surf < reac_state[1]):
                        p_prod = 0
                    else:
                        continue
                    if p_prod == 0: # Need to check if this came back from product (long term recrossing)
                        recrossed = self._check_recross(traj[idx:idx+frame_delay], 
                                                        id, ir_extreme_zs,
                                                        prod_state)
                        if recrossed:
                            continue
                counter += 1
                p_reacXprod += p_reac*p_prod    
            num_avg = p_reacXprod/counter
            time_sum_of_num_avg += num_avg
        time_avg = time_sum_of_num_avg/len(traj_for_use)
        return 1 - time_avg

    def plot_time_corr(self, time_lim, n_divisions, reac_states, prod_states, labels):
        x_axis = np.arange(0, time_lim, int(time_lim)/n_divisions)
        fig, ax = plt.subplots()
        for reac_state, prod_state, label in zip(reac_states, prod_states, labels):
            Y_set = np.zeros(shape=(len(self.trajs), len(x_axis)))
            for i, traj in enumerate(self.trajs):
                print(f"Calculating for: reac_state, {reac_state}, trajectory {i}")
                Y_set[i] = [self._calc_time_corr(traj = traj, time_delay=x, 
                                                 reac_state=reac_state, 
                                                 prod_state=prod_state) 
                                                 for x in x_axis]
            y_axis = np.mean(Y_set, axis=0)
            np.savetxt(f'{label[1]}.txt', np.vstack((x_axis, y_axis)).T, fmt='%1.3f', delimiter=', ')
            ax.plot(x_axis, y_axis, marker='.', label=label)
            ax.set(xlabel="$t,\ delay\ in\ ps$", ylabel="$C(t)$")
            ax.legend()
            # ax.set_ylim(0.0,1.01)
        plt.savefig("tc_long.png", dpi=300, bbox_inches='tight')
# %%
def main():
    # path_to_sim = Path("/work/surf/H_movement")
    path_to_sim = Path("/work/surf/coverage/mirror/n_O") 
    sim_prefix = "sim_"
    sim_indices = [0,3]
    sim_paths, init_struc_paths = [], []
    for ind in range(sim_indices[0], 1+sim_indices[1], 1):
        sim_paths.append(path_to_sim/(sim_prefix+str(ind))/"concat.traj")
        init_struc_paths.append(path_to_sim/(sim_prefix+str(ind))/"mix.xyz")

    TC = time_corr(sim_paths=sim_paths,
                   init_struc_paths=init_struc_paths,
                   skip_init_time=10.0,
                   skip_traj_frames=100 # original dt = 0.1 ps, 10 frames=1ps
                   )
    reac_states=[[2.6, 3.75], [5.75, 6.85], [15.0, 16.15]]
    transition_region = 1.0 #angstroms
    prod_states=[[reac_states[0][0]-transition_region, reac_states[0][1]+transition_region],
                 [reac_states[1][0]-transition_region, reac_states[1][1]+transition_region], 
                 [reac_states[2][0]-transition_region, reac_states[2][1]+transition_region]]
    TC.plot_time_corr(time_lim = 300.0,
                      n_divisions=30,
                      reac_states=reac_states,
                      prod_states=prod_states, # one number means above that is product, list of two means outside this range is product
                      labels = ["$1^{st}\ layer$", "$2^{nd}\ layer$", "$Bulk$"]
                      )

if __name__ == "__main__":
    main()
# %%