import random, time
import numpy as np
import glob, os
from pathlib import Path

from ase.io.lammpsdata import write_lammps_data
from ase.io import read, write
from ase.visualize import view
from ase import Atoms

def create_mix(water_box_files, slab_file, adsorbates_file=None):
    '''
    Puts surface adsorbate molecules in random cus sites of slab
    Puts O/OH on all bridge site randomly
    Adds water box of 400 molecules on top
    water_box_files: [] list of .xyz files. One random file will be chosen for mix.
    adsorbates_file: ""; slab_file: "" single structure
    '''
    #adsorbates = read(adsorbates_file, ":")
    slab = read(slab_file)
    water = read(random.choice(water_box_files))
    '''
    #Cus site additions
    max_Ir_Z = max([atom.z for atom in slab if atom.symbol == 'Ir'])
    Ir_bri = np.array([atom.index for atom in slab if (atom.symbol == 'Ir') and (atom.z == max_Ir_Z)])
    Ir_cus = np.array([atom.index for atom in slab if (atom.symbol == 'Ir' and (np.round(max_Ir_Z - atom.z, 4) == 0.0953) and atom.index not in Ir_bri)])
    cus = slab[Ir_cus]
    oxy_height = 2.0
    cus_sites = [pos+[0,0,oxy_height] for pos in cus.get_positions()]
    layer1 = Atoms()
    for site in cus_sites:
        random_adsorbate = random.choice(adsorbates).copy()
        random_adsorbate.positions += site
        layer1 += random_adsorbate
    #Bri site additions
    O_bri = np.array([atom.index for atom in slab if (atom.symbol == 'O' and atom.z > max_Ir_Z and min(slab.get_distances(atom.index, Ir_bri, mic=True))<2.3 and min(slab.get_distances(atom.index, Ir_cus, mic=True))>2.3)])
    bri = slab[O_bri]
    hydro_height = 0.96
    bri_sites = [pos+[0,0,hydro_height] for pos in bri.get_positions()]
    layer2 = Atoms()
    for site in bri_sites:
        H_atom = Atoms('H', positions=[(0, 0, 0)])
        Null = Atoms()
        random_bri = random.choice([H_atom, Null]).copy()
        random_bri.positions += site
        layer2 += random_bri
    '''
    coverage = slab #+ layer1 + layer2
    og_cell = water.get_cell()
    og_lx = og_cell[0][0]
    og_ly = og_cell[1][1]
    og_lz = og_cell[2][2]
    zs = [arr[-1] for arr in coverage.get_positions()]
    thickness = max(zs) -  min(zs)
    new_cell = np.add([og_lx,og_ly,og_lz], [0,0,1.0+thickness])
    pos0 = water.get_positions()
    pos1 = [np.add(pos,[0,0,1.50+thickness]) for pos in pos0]
    water.set_positions(pos1)
    water.set_cell(new_cell)
    mix = water+coverage
    vacuum = 30.0
    mix.cell[2][-1] += vacuum
    return mix

# Empty surface:
sim_indexes         = int(input("Provide initial index: ")), int(input("Provide final index: "))
cwd                 = Path(os.getcwd())
run_file            = cwd/"run.sh"
lmps_template_file  = cwd/"template.inp"

water_box_files     = glob.glob(str(cwd/"../W400_boxes/*xyz"))
#adsorbate_file      = "/ptmp/nbapat/surf/coverage/rand_cov_iter3/adsorbates.xyz"
slab_file           = cwd/"pristinex4.xyz"

for i in range(sim_indexes[0],sim_indexes[1],1):
    Path(cwd/f"sim_{i}").mkdir(parents=False, exist_ok=False)
    path_to_sim = cwd/f"sim_{i}"
    # Writing a random starting structure to sim folder
    mix = create_mix(water_box_files=water_box_files,
                     slab_file=slab_file)
    write(path_to_sim / "mix.xyz", mix)
    write_lammps_data(path_to_sim / "mix.lmps", mix, atom_style='full')
    # Writing lammps input in sim folder with random seed
    random.seed(time.time())
    with open(lmps_template_file, "rt") as template:
        data_lmps = template.read()
        data_lmps = data_lmps.replace('SEED', str(random.randint(1,100000)))
    with open(path_to_sim/"lammps.inp", "wt") as new_lmps:
        new_lmps.write(data_lmps)
    # Copy run.sh file to simulation folder
    with open(run_file, "rt") as runner:
        data_run = runner.read()
        data_run = data_run.replace('JOB_NAME', f"sim_{i}")
    with open(path_to_sim/"run.sh", "wt") as new_runner:
        new_runner.write(data_run)
