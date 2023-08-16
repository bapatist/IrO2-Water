from ase.io import read, write
from ase.constraints import FixAtoms
from ase.neb import NEB

initial=read('left.xyz')
final=read('right.xyz')

constraint=FixAtoms(mask=[atom.tag > 0 for atom in initial])

Nimages=8

images=[initial]
for i in range(Nimages):
        image=initial.copy()
        image.set_constraint(constraint)
        images.append(image)

images.append(final)
neb=NEB(images,climb=True)
neb.interpolate('idpp', apply_constraint=True)

cnt=0
#Check interpolated structures!
write("initial_path.xyz", images)
for i in range(Nimages+2):
    write("images"+str(i+1)+".png",images[i])