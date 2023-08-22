from ase.io import read, write
from ase.io.trajectory import Trajectory
import pandas as pd
from pathlib import Path

def fix_md(dir_prefix, sim_indices):
    try:
        traj_files = list(Path(".").glob("**/*.traj"))
        if len(traj_files) > 0:
            raise ValueError(f'A traj file already exists {traj_files}')
    except FileNotFoundError:
        pass
    finally:
        out_traj_all = Trajectory(f"concat.traj", 'w')
    for ind in range(sim_indices[0], 1+sim_indices[1], 1):
        print(f"Fixing sim index: {ind}")
        workdir = f"{dir_prefix}{ind}"
        out_traj = Trajectory(f"{workdir}/movie.traj", 'w')
        md_og = read(f"{workdir}/movie.xyz", ":")
        for image in md_og:
            image.info = {}
        log = pd.read_table(f"{workdir}/logCut.lammps", delim_whitespace=True)
        starting_struc = read(f"mix.xyz")
        cell = starting_struc.cell.cellpar()
        if len(log) != len(md_og):
            raise ValueError(f'log has {len(log)} entries while md has {len(md_og)} images')
        for i, image in enumerate(md_og):
            if i==0 and ind>0: # To avoid repeats at the joining points
                continue
            image.set_cell(cell)
            image.set_pbc(True)
            image.info['time'] = log['Time'][i]
            image.info['temp'] = log['Temp'][i]
            image.info['md_energy']  = log['TotEng'][i]
            image.info['pot_energy'] = log['PotEng'][i]
            out_traj.write(image)
            out_traj_all.write(image)
    concat_traj = Trajectory("concat.traj", "r")
    print(f"Concatanated trajectory of length {len(concat_traj)} from {concat_traj[0].info['time'], concat_traj[-1].info['time']} ps")
#        os.remove(f"{workdir}/movie.xyz")

def main():
    sim_indices = int(input("Provide initial index: ")), int(input("Provide final index: "))
    fix_md(dir_prefix="", sim_indices=sim_indices)

if __name__ == '__main__':
    main()
