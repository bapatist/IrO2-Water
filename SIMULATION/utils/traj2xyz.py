from ase.io.trajectory import Trajectory
from ase.io import write

traj_file = input("Input full traj file name:")

start, end = [int(x) for x in input("Enter start and end frame index: ").split()]
traj = Trajectory(traj_file)[start:end]

skip_frames = int(input("Skip every x frames?:"))
write(f"{traj_file[:-4]}xyz", traj[::skip_frames])
