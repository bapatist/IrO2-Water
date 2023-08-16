import quippy
from ase.io import read, write

def set_ener_forces(atoms_list, potential):
    for atoms in atoms_list:
        atoms.set_calculator(potential)
        atoms.arrays['GAP_forces'] = atoms.get_forces()
        atoms.info['GAP_energy'] = atoms.get_potential_energy()
    return None

strucs = read("./mep_latest.xyz", ":")
pot = quippy.potential.Potential(param_filename='/ptmp/nbapat/surf/ITER4/GAP/GAP_21.xml')

set_ener_forces(strucs, pot)
write("./mep_gap.xyz", strucs)