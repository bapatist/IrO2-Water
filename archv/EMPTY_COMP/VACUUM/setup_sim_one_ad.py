# %%
from dataclasses import dataclass
import random, time
import numpy as np
import glob, os
from pathlib import Path

from ase.io.lammpsdata import write_lammps_data
from ase.io import read, write
from ase import Atoms
# %%
@dataclass
class SimSetter():
    adsorbates_file: Path
    slab_file: Path
    water_boxes_files: list
    def __post_init__(self) -> None:
        self.adsorbates = read(self.adsorbates_file, ":")
        self.slab = read(self.slab_file)
        self.water = read(random.choice(self.water_boxes_files))

    @staticmethod
    def prep_cus_site_ids(nsites: int, nOH=0, nH2O=0, nOOH=0, nO=0) -> list:
        assert nOH + nH2O + nOOH + nO < nsites, "Ungli pakad ke pahuncha na pakde"
        ids = np.full(nsites, None)
        id_OH  = np.random.choice(list(range(nsites)), nOH,  replace=False)
        avail_sites = [site for site in list(range(nsites)) if site not in id_OH]
        id_H2O = np.random.choice(avail_sites, nH2O, replace=False)
        avail_sites = [site for site in avail_sites if site not in id_H2O]
        id_OOH = np.random.choice(avail_sites, nOOH, replace=False)
        avail_sites = [site for site in avail_sites if site not in id_OOH]
        id_O   = np.random.choice(avail_sites, nO,   replace=False)
        np.put(ids, np.hstack([id_OH, id_H2O, id_OOH, id_O]), np.hstack([nOH*['OH'], nH2O*['H2O'], nOOH*['OOH'], nO*['O']]))
        return ids

    @staticmethod
    def prep_bri_site_ids(nsites:int, nOH=0) -> list:
        assert nOH < nsites, "Ungli pakad ke pahuncha na pakde"
        ids = np.full(nsites, None)
        id_OH = np.random.choice(list(range(nsites)), nOH, replace=False)
        np.put(ids, id_OH, nOH*['OH'])
        return ids

    def add_cover(self, cus_site_ids, bri_site_ids=None):
        max_Ir_Z = max([atom.z for atom in self.slab if atom.symbol == 'Ir'])
        Ir_bri = np.array([atom.index for atom in self.slab if (atom.symbol == 'Ir') and (atom.z == max_Ir_Z)])
        Ir_cus = np.array([atom.index for atom in self.slab if (atom.symbol == 'Ir' and (np.round(max_Ir_Z - atom.z, 4) == 0.0953) and atom.index not in Ir_bri)])
        cus = self.slab[Ir_cus]
        oxy_height = 2.0
        cus_sites = [pos+[0,0,oxy_height] for pos in cus.get_positions()]
        assert len(cus_sites) == len(cus_site_ids), "No. of cus-sites identified are not equal to no. of cus-site IDs"
        layer1 = Atoms()
        for site_pos, site_id in zip(cus_sites, cus_site_ids):
            if site_id == None:
                continue
            elif site_id == 'OH':
                ad_copy = [ad for ad in self.adsorbates if ad.get_chemical_formula()=='HO'][0].copy()
                ad_copy.positions += site_pos
                layer1 += ad_copy
            elif site_id == 'H2O':
                ad_copy = [ad for ad in self.adsorbates if ad.get_chemical_formula()=='H2O'][0].copy()
                ad_copy.positions += site_pos
                layer1 += ad_copy
            elif site_id == 'OOH':
                ad_copy = [ad for ad in self.adsorbates if ad.get_chemical_formula()=='HO2'][0].copy()
                ad_copy.positions += site_pos
                layer1 += ad_copy
            elif site_id == 'O':
                ad_copy = [ad for ad in self.adsorbates if ad.get_chemical_formula()=='O'][0].copy()
                ad_copy.positions += site_pos
                layer1 += ad_copy
        layer2 = Atoms()
        if bri_site_ids is not None:
            assert len(bri_site_ids) == len(cus_site_ids), "Cus & Bridge sites should be equal in number"
            O_bri = np.array([atom.index for atom in self.slab if
                              (atom.symbol == 'O' and atom.z > max_Ir_Z and
                               min(self.slab.get_distances(atom.index, Ir_bri, mic=True))<2.3 and
                               min(self.slab.get_distances(atom.index, Ir_cus, mic=True))>2.3)])
            bri = self.slab[O_bri]
            hydro_height = 0.96
            bri_sites = [pos+[0,0,hydro_height] for pos in bri.get_positions()]
            for site_pos, site_id in zip(bri_sites, bri_site_ids):
                if site_id==None:
                    continue
                else:
                    H_atom = Atoms('H', positions=[(0, 0, 0)])
                    H_atom.positions += site_pos
                    layer2 += H_atom
        coverage = self.slab + layer1 + layer2
        # write("coverage_in_vacuum.xyz", coverage)
        return(coverage)

    def add_water(self, slab_cov, vacuum=30.0):
            water_cell = self.water.get_cell()
            water_lx, water_ly, water_lz = water_cell[0][0], water_cell[1][1], water_cell[2][2]
            zs = [arr[-1] for arr in slab_cov.get_positions()]
            thickness = max(zs) -  min(zs)
            new_cell = np.add([water_lx, water_ly, water_lz], [0,0,1.0+thickness])
            pos0 = self.water.get_positions()
            pos1 = [np.add(pos,[0,0,0.75+thickness]) for pos in pos0]
            self.water.set_positions(pos1)
            self.water.set_cell(new_cell)
            mix = self.water + slab_cov
            mix.cell[2][-1] += vacuum
            # write("mix.xyz", mix)
            return mix
# %%
def main(job_type="sim"):
    sim_indexes         = int(input("Provide initial index: ")), int(input("Provide final index: "))
    cwd                 = Path(os.getcwd())
    run_file            = cwd/"run.sh"
    lmps_template_file  = cwd/"template.inp"

    water_box_files     = glob.glob("/ptmp/nbapat/surf/coverage/W400_boxes/*xyz")
    adsorbates_file      = "/ptmp/nbapat/surf/coverage/adsorbates.xyz"
    slab_file           = "/ptmp/nbapat/surf/coverage/pristinex4.xyz"

    for i in range(sim_indexes[0],sim_indexes[1],1):
        folder = f"sim_{i}"
        Path(cwd/folder).mkdir(parents=False, exist_ok=False)
        path_to_sim = cwd/folder
        setup = SimSetter(water_boxes_files=water_box_files,
                        adsorbates_file=adsorbates_file,
                        slab_file=slab_file)
        bri_ids  = setup.prep_bri_site_ids(nsites=24, nOH=0)
        cus_ids  = setup.prep_cus_site_ids(nsites=24, nOH=0, nH2O=0, nOOH=0, nO=0)
        slab_cov = setup.add_cover(cus_site_ids=cus_ids, bri_site_ids=bri_ids)
        #mix      = setup.add_water(slab_cov=slab_cov)

        write(path_to_sim / "mix.xyz", slab_cov)
        write_lammps_data(path_to_sim / "mix.lmps", slab_cov, atom_style='full')

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
            data_run = data_run.replace('JOB_NAME', f"{job_type}_{i}")
        with open(path_to_sim/"run.sh", "wt") as new_runner:
            new_runner.write(data_run)

if __name__ == "__main__":
    main(job_type="V_emp")
#%%
