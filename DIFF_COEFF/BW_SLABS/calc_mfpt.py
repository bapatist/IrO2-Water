# Compute Position-dependent diffusion from MFPT. Based on PRE 106, 014103

from ase.io import read, write
import numpy as np
import sys, time
from diff_mfpt import calc_mfpt_prob


traj = read("output.traj", ':')
data = []

params = [
(8, 4000),
(10, 4000),
(16, 1000),
(20, 800),
]
    

for nbins, subLen in params:

    binCenter, mfpt, probArr = calc_mfpt_prob(traj, 2, 0.01, subLen, [0,20], nbins)
    
    print(f"Mean First Passage Time:\n{mfpt}")
    print(f"Probability:\n{probArr}")
    
    L = 0.5*(binCenter[1]-binCenter[0])
    #D = L**2/mfpt[0]*(probArr[0,0]-probArr[0,1])/np.log(probArr[0,0]/probArr[0,1])/1000
    D = L**2/mfpt/2/1000
    print(f"Diffusion coefficient: {D.mean()} A2/fs")
    print(f"Diffusion coefficient err: {D.std()} A2/fs")
    print("\n")
    
    data.append([L, mfpt.mean(), D.mean(), D.std()])

np.savetxt('mfpt.dat', data, fmt='%.8f')
