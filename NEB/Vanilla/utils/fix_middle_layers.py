import numpy as np
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.visualize import view

atoms = read("./slab.xyz")

# Create a boolean array for tagging the middle two layers
tagged_atoms = np.zeros(len(atoms), dtype=bool)
for i, atom in enumerate(atoms):
    if  atom.z < 14.5:
        tagged_atoms[i] = True

# Apply the tags to the ASE Atoms object
atoms.set_tags(tagged_atoms)

# Create constraints to fix the tagged atoms
constraints = [FixAtoms(mask=tagged_atoms)]

# Apply the constraints to the ASE Atoms object
atoms.set_constraint(constraints)
write("slab_fixed.xyz", atoms)
# Now 'atoms' contains the tagged and fixed structure