#%%
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from ase.io.trajectory import Trajectory
from ase.io import read

import matplotlib
font = {'size'   : 18}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
#%%
@dataclass
class time_corr():
    skip_init_time: float #skip initial part of MD
    skip_traj_frames: int #hopping these frames
    iridium_surf_Zs: list #top and bottom surface iridium Z
    init_struc_paths: list
    sim_paths: list

    @property
    def init_struc_paths(self):
        return self._init_struc_paths
    
    @init_struc_paths.setter
    def init_struc_paths(self, init_struc_paths):
        _mixes = []
        for _mix in init_struc_paths:
            _mixes.append(read(_mix))
        self.init_strucs = _mixes

    @property
    def sim_paths(self):
        return self._sim_paths

    @sim_paths.setter
    def sim_paths(self, sim_paths):
        _trajs = []
        for _path in sim_paths:
            _trajs.append(Trajectory(_path)[::self.skip_traj_frames])
        self.dt = np.round(_trajs[0][1].info['time'] - _trajs[0][0].info['time'], 2)
        skip_init_frames = int(self.skip_init_time/self.dt)
        self.trajs = [traj[skip_init_frames:] for traj in _trajs]
        print("dt = ", self.dt, "ps")
    
    def _id_cus_oxy(self, init_mix):
        O_cus = np.array([atom.index for atom in init_mix if
                          atom.symbol == 'O' and
                          (np.round(atom.z, 2) in 
                            [np.round(self.iridium_surf_Zs[0]+1.9908, 2), 
                             np.round(self.iridium_surf_Zs[1]-1.9908, 2)])]) # Starting positions
        # print(f"{int(len(O_cus))} oxygens have been ID'd at the surfaces")
        return O_cus
    
    def _id_H_interface(self, im):
        id_H_interface = np.array([atom.index for atom in im if 
                                   atom.symbol == 'H' and
                                   (atom.z < self.iridium_surf_Zs[0]+4.0 or
                                    atom.z > self.iridium_surf_Zs[1]-4.0)])
        return id_H_interface
    
    def _id_OH_reac(self, im, id_oxys, id_H_interface):
        id_OH_reactant = [atom.index for atom in im if
                          atom.index in id_oxys and 
                          len([d for d in im.get_distances(atom.index, 
                                                           id_H_interface, 
                                                           mic=True) 
                               if d<=1.1])==1 and
                          len([d for d in im.get_distances(atom.index, 
                                                           id_H_interface, 
                                                           mic=True) 
                               if d<=1.4])==1]
        # print(f"Found {int(len(id_OH_reactant))} reactant OH at time {im.info['time']}")
        return id_OH_reactant
    
    def _count_neighs_suburbs(self, image, id_O, id_H):
        n_neigh  = len([d for d in image.get_distances(id_O, id_H, mic=True) if d<=1.1])
        n_suburb = len([d for d in image.get_distances(id_O, id_H, mic=True) if 1.1<d<1.4])
        return n_neigh, n_suburb
    
    def _calc_time_corr(self, init_mix, traj, time_delay, prod_state):
        frame_delay = int(time_delay/self.dt)
        if frame_delay==0:
            traj_for_use = traj
        else:
            traj_for_use = traj[:-frame_delay]
        time_sum_of_num_avg = 0
        id_oxys = self._id_cus_oxy(init_mix=init_mix)
        for idx, im_reac in enumerate(traj_for_use):
            # Identifying oxys in OH state at t=0
            id_H_inter_reac  = self._id_H_interface(im=im_reac)
            id_OH_reactants = self._id_OH_reac(im=im_reac, id_oxys=id_oxys, 
                                               id_H_interface=id_H_inter_reac)
            p_reac, p_prod = 1, 0
            p_reacXprod, counter = 0, 0
            im_prod = traj[idx+frame_delay]
            id_H_inter_prod = self._id_H_interface(im=im_prod)
            for id_O in id_OH_reactants:
                n_neigh, n_suburb = self._count_neighs_suburbs(image=im_prod, 
                                                                   id_O=id_O,
                                                                   id_H=id_H_inter_prod)
                if prod_state == 'H2O':
                    if n_neigh==0 or n_suburb>0:
                        continue
                    elif n_neigh==2 and n_suburb==0:
                        p_prod = 1
                    else:
                        p_prod = 0
                    counter += 1
                elif prod_state == 'O':
                    if n_neigh>1 or n_suburb>0:
                        continue
                    elif n_neigh==0 and n_suburb==0:
                        p_prod = 1
                    else:
                        p_prod = 0
                    counter += 1
                p_reacXprod += p_reac*p_prod
            num_avg = p_reacXprod/counter
            time_sum_of_num_avg += num_avg
        time_avg = time_sum_of_num_avg/len(traj_for_use)
        return 1 - time_avg

    def plot_time_corr(self, time_lim, prod_states, labels):
        x_axis = np.arange(0, time_lim, int(time_lim)/10) #self.dt)
        fig, ax = plt.subplots()
        Y_set = np.zeros(shape=(len(self.trajs), len(x_axis)))
        for prod_state, label in zip(prod_states, labels):
            for i, (traj, mix) in enumerate(zip(self.trajs, self.init_strucs)):
                Y_set[i] = [self._calc_time_corr(init_mix=mix,
                                                traj = traj,
                                                time_delay=x,
                                                prod_state=prod_state)
                                                for x in x_axis]
            y_axis = np.mean(Y_set, axis=0)
            ax.plot(x_axis, y_axis, marker='.', label=label)
            ax.set(xlabel="$t,\ delay\ in\ ps$", ylabel="$C(t)$")
            ax.legend()
            # ax.set_ylim(0.6,1.01)
            plt.savefig("tc_oh_h2o.png", dpi=300, bbox_inches='tight')
# %%
def main():
    sim_prefix = "sim_"
    sim_indices = [0,3]
    sim_paths, init_struc_paths = [], []
    for ind in range(sim_indices[0], 1+sim_indices[1], 1):
        sim_paths.append(f"./{sim_prefix}{ind}/concat.traj")
        init_struc_paths.append(f"./{sim_prefix}{ind}/mix.xyz")

    TC = time_corr(sim_paths=sim_paths,
                   init_struc_paths=init_struc_paths,
                   iridium_surf_Zs = [4.39915211, 41.15683543],
                   skip_init_time=0.0,
                   skip_traj_frames=50 # original dt = 0.1 ps, 10 frames=1ps
                   )
    TC.plot_time_corr(time_lim=700, prod_states=["O", "H2O"], 
                    #   labels=["$OH\ to\ H_{2}O$"])
                      labels=["$OH\ to\ O$", "$OH\ to\ H_{2}O$"])

if __name__ == "__main__":
    main()
# %%
