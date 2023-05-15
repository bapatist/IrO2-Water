# Compute Position-dependent diffusion from MFPT. Based on PRE 106, 014103

from ase.io import read, write
import numpy as np
import sys, time
from diff_mfpt import calc_mfpt_prob

#slab_ends     = [4.39915211, 41.15683543]
slab_ends = [4.39915211, 24.0]

trajs, id_water_oxys = [], []
for file in sys.argv[1:]:
    print(f"Reading {file} as trajectory")
    traj = read(file, ":")[1000:]
    id_water_oxys.append([atom.index for atom in traj[0] if
                          atom.symbol=='O' and
                          slab_ends[0] < atom.z < slab_ends[1]])
    trajs.append(traj)

data         = []
nbins        = 51
sub_traj_len = 200

sample_range = [slab_ends[0]+0.5, slab_ends[1]-0.5]

for ind, (traj, ids) in enumerate(zip(trajs, id_water_oxys)):

    binCenter, mfpt, probArr = calc_mfpt_prob(traj=traj, axis=2, timestep=0.1,
                                              subLen=sub_traj_len,
                                              sampleRange=sample_range, nbins=nbins,
                                              indices=ids)

    print(f"Mean First Passage Time:\n{mfpt}")
    print(f"Probability:\n{probArr}")

    L = 0.5*(binCenter[1]-binCenter[0])
    #D = L**2/mfpt[0]*(probArr[0,0]-probArr[0,1])/np.log(probArr[0,0]/probArr[0,1])/1000
    #D = L**2/mfpt/2/1000
    D = []
    for i in range(nbins):
        if abs(probArr[i,0]-probArr[i,1]) > 1e-5:
            D.append(L**2/mfpt[i]*(probArr[i,0]-probArr[i,1])/np.log(probArr[i,0]/probArr[i,1])/1000)
        else:
            D.append(L**2/mfpt[i]/2000)
    D = np.array(D)
    D = [d*1000 for d in D] # converting A2/fs to A2/ps
    binCenter = [bc-slab_ends[0] for bc in binCenter] # referencing zero to bottom slab end
    print("Positions of bins:")
    print(binCenter)
    print("Diffusion coefficient in A2/ps:")
    print(D)
    print("\n")

    data = np.array([binCenter, D]).T
    np.savetxt(f'diff_{ind}.dat', data, fmt='%.8f')
    np.savetxt(f'mfpt_{ind}.dat', mfpt, fmt='%.8f')
    np.savetxt(f'prob_{ind}.dat', probArr, fmt='%.8f')
