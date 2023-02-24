from ase.io import read, write
from ase.io.trajectory import Trajectory
import pandas as pd
import os

def fix_md(sim_indices):
    for ind in range(sim_indices[0], sim_indices[1], 1):
        print(f"Fixing sim index: {ind}")
        workdir = f"sim_{ind}"
        try:
            out_traj = Trajectory(f"{workdir}/newmd.traj", 'r')
            if len(out_traj) > 0:
                raise ValueError(f'A traj file already exists: {workdir}/newmd.traj containing {len(out_traj)} images')
        except FileNotFoundError:
            pass
        finally:
            out_traj = Trajectory(f"{workdir}/newmd.traj", 'w')
        md_og = read(f"{workdir}/min_md.xyz", ":")
        for image in md_og:
            image.info = {}
        log = pd.read_table(f"{workdir}/logCut.lammps", delim_whitespace=True)
        starting_struc = read(f"{workdir}/mix.xyz")
        cell = starting_struc.cell.cellpar()
        if len(log) != len(md_og):
            raise ValueError(f'log has {len(log)} entries while md has {len(md_og)} images')
        for i, image in enumerate(md_og):
            image.set_cell(cell)
            image.info['density'] = log['Density'][i]
            image.set_pbc(True)
            image.info['time'] = log['Time'][i]
            image.info['temp'] = log['Temp'][i]
            image.info['md_energy'] = log['TotEng'][i]
            out_traj.write(image)
        os.remove(f"{workdir}/min_md.xyz")

def main():
        sim_indices = int(input("Provide initial index: ")), int(input("Provide final index: "))
        fix_md(sim_indices=sim_indices)

if __name__ == '__main__':
    main()

