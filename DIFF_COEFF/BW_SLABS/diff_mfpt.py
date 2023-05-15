# Compute Position-dependent diffusion from MFPT. Based on PRE 106, 014103

from ase.io import read, write
import numpy as np
import sys, time

def calc_mfpt_prob(traj, axis, timestep, subLen, sampleRange, nbins, indices=None, pbc=True):
    """
    Get MFPT and passage probability.

    Params:
    traj: ASE trajectory, in which atomic positions should be unwrapped.
    Assume the simulation cell is orthogonal and doesn't change in trajectory.
    axix: 0, 1 or 2, determine the axis to analyse
    timestep: time interval between neighboring structures in traj, in ps unit
    subLen: int, traj will be seperated to short trajectories containing subLen structures to sample first passage events.
    sampleRange: list or tuple, like [a,b], determine the spatial range to sample along axis, in AA
    nbins: int, number of bins in sampleRange
    indices: list of ints, indices of atoms to be sampled. If None, all atoms are consider
    pbc: bool, If True, digitize distances based on scaled positions under PBC. The traj should has cell. If False, digitize distances based on Cartesian positions.

    Outputs:

    """

    minD, maxD = sampleRange
    assert minD < maxD, "Set correct sampleRange!"
    cellpar = traj[0].cell.cellpar()[:3]
    assert minD >= 0 and maxD <=cellpar[axis], "Set corret sampleRange or check cell!"
    bins = np.linspace(minD, maxD, num=nbins+1)
    L = (bins[1] - bins[0])/2 # Half bin size, L in PRE 106, 014103
    binCenter = (bins[1:] + bins[:-1])/2
    # To save all FPT in every bin.
    # fptArr[:,n] means number of passage events with FPT of (n+1/2)*timestep
    # fptArr[:,-1] records all survival events (FPT = inf)
    fptArr = np.zeros((nbins, subLen))
    # To record first passage event for two directions.
    # passArr[:,0]: x -> x-L; probArr[:,1]: x -> x+L. relative to P- and P+
    passArr = np.zeros((nbins, 2))
    nAtom = len(traj[0])
    if indices is None:
        atInd = range(nAtom)
    else:
        atInd = indices[:]
    nCfg = len(traj)
    nSubTrj = nCfg//subLen # number of short trajectories
    nSample = 0 # To record total number of atoms to be considered

    print(f"Number of bins: {nbins}")
    print(f"Half bin size L: {L} AA")
    print(f"Sub trajectory length: {timestep*subLen} ps")

    for i in range(nSubTrj):
        print(f"\nSub trajectory {i} ...")
        subTrj = traj[i*subLen:(i+1)*subLen]
        initAts = subTrj[0]
        # Determine indices of atoms which are in the bin initially.
        initD = initAts.get_positions()[:,axis]
        if pbc:
            # Use scaled positions to consider PBC
            sclD = initAts.get_scaled_positions()[:,axis]
            sclBins = bins/cellpar[axis]
            digRes = np.digitize(sclD, sclBins) # Bin index for sampled atoms. Note: atoms might fall outside sampleRange
        else:
            digRes = np.digitize(initD, bins)
        sampleInd = [index for index in atInd if 0 < digRes[index] <= nbins]
        nSample += len(sampleInd)
        print(f"Tracking {len(sampleInd)} atoms. Totaly {nSample} atoms.")
        # Check first passage event
        outInd = [] # atoms which has escaped
        for step, ats in enumerate(subTrj[1:]):
            # currently survived atoms
            curInd = [index for index in sampleInd if index not in outInd]
            if len(curInd) == 0:
                break
            curInd = np.array(curInd)
            curD = ats.get_positions()[curInd,axis]

            # slower way to get FPT
            # minusInd = []
            # plusInd = []
            # for m, index in enumerate(curInd):
            #     if curD[m] < bins[digRes[index]-1]:
            #         minusInd.append(index)
            #     elif curD[m] > bins[digRes[index]]:
            #         plusInd.append(index)


            # Get distance difference
            diffD = curD - initD[curInd]
            minusInd = curInd[np.nonzero(diffD<-1*L)[0]]
            plusInd = curInd[np.nonzero(diffD>L)[0]]

            # Count passage events
            for index in minusInd:
                fptArr[digRes[index]-1,step] += 1
                passArr[digRes[index]-1,0] += 1
                outInd.append(index)
            for index in plusInd:
                fptArr[digRes[index]-1,step] += 1
                passArr[digRes[index]-1,1] += 1
                outInd.append(index)
        print("Totally Detected:")
        print(f"{int(passArr[:,0].sum())} minus first passage events")
        print(f"{int(passArr[:,1].sum())} plus first passage events")
        # If there are still survival...
        if len(outInd) < len(sampleInd):
            curInd = [index for index in sampleInd if index not in outInd]
            print(f"Note: there are {len(curInd)} remaining atoms in sub trajectory {i}.")
            for index in plusInd:
                fptArr[digRes[index]-1,-1] += 1
    # Number of surviving atoms
    nEvent = int(passArr.sum())
    print(f"\nTotally track {nSample}, sample {nEvent} events, {nSample-nEvent} surving events.")
    # Compute MFPT in every bin, (n+1/2)*\delta t; Ignore surviving atoms
    mfpt = (fptArr[:,:-1] * (np.arange(subLen-1) + 0.5)).sum(1)/fptArr[:,:-1].sum(1)
    # The previous mfpt is in unit of step, now transform it to ps
    mfpt *= timestep

    # Compute probability
    probArr = passArr/np.expand_dims(passArr.sum(1),1)


    return binCenter, mfpt, probArr


# traj = read("../MBD_5-1370K-2.xyz", ':')
# binCenter, mfpt, probArr = calc_mfpt_prob(traj, 2, 0.1, 200, [30,40], 5, pbc=False)


