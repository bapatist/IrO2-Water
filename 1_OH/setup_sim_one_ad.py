# %%
from dataclasses import dataclass
import random, time
import numpy as np
import glob
from pathlib import Path

from ase.io.lammpsdata import write_lammps_data
from ase.io import read, write
from ase.visualize import view
from ase import Atoms
# %%
@dataclass
class SimSetter():
    nsims: int
    run_sh_file: Path
    lmps_template_file: Path
    water_boxes_files: list
    adsorbates_file: Path
    slab_file: Path
    def __post_init__(self) -> None:
        self.adsorbates = read(self.adsorbates_file, ":")
        self.slab = read(self.slab_file)
        self.water = read(random.choice(self.water_boxes_files))
    
    @staticmethod
    def prep_site_ids(nsites: int, nOH=0, nH2O=0, nOOH=0, nO=0) -> list:
        if nOH + nH2O + nOOH + nO > nsites:
            raise ValueError("nOH + nH2O + nOOH + nO > nsites")
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
    
    def add_cover(self, cus_site_ids, bri_site_ids=None):
        # Cus site additions
        max_Ir_Z = max([atom.z for atom in self.slab if atom.symbol == 'Ir'])
        Ir_bri = np.array([atom.index for atom in self.slab if (atom.symbol == 'Ir') and (atom.z == max_Ir_Z)])
        Ir_cus = np.array([atom.index for atom in self.slab if (atom.symbol == 'Ir' and (np.round(max_Ir_Z - atom.z, 4) == 0.0953) and atom.index not in Ir_bri)])
        cus = self.slab[Ir_cus]
        oxy_height = 2.0
        cus_sites = [pos+[0,0,oxy_height] for pos in cus.get_positions()]
        if len(cus_sites) != len(cus_site_ids):
            raise ValueError("No. of cus-sites identified are not equal to no. of cus-site IDs")
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
        coverage = self.slab + layer1
        write("coverage_in_vacuum.xyz", coverage)
        return(coverage)
# %%
setup = SimSetter(nsims=1, run_sh_file=Path("./run.sh"),
                  lmps_template_file=Path("./template.inp"),
                  water_boxes_files=glob.glob("../../W400_boxes/*.xyz"),
                  adsorbates_file=Path("./adsorbates.xyz"),
                  slab_file=Path("./pristinex4.xyz"))
cus_ids = setup.prep_site_ids(nsites=24, nOH=2, nH2O=1, nOOH=2, nO=1)
setup.add_cover(cus_site_ids=cus_ids)
#%%