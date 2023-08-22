from ase.io import read, write
from ase.constraints import FixAtoms
from ase.neb import NEB
from ase.calculators.vasp import Vasp
import yaml
import copy
from gpatom.aidneb import AIDNEB

initial=read('left_opt.xyz')
final=read('right_opt.xyz')

constraint=FixAtoms(mask=[atom.tag > 0 for atom in initial])

Nimages=8

images=[initial]
for i in range(Nimages):
    image=initial.copy()
    image.set_constraint(constraint)
    images.append(image)

images.append(final)
images[-1].set_constraint(constraint)
neb=NEB(images,climb=True)
neb.interpolate('idpp', apply_constraint=True)

#Check interpolated structures!
write("initial_path.xyz", images)

# for i in range(Nimages+2):
#     write("images"+str(i+1)+".png",images[i])

params_file = "vasp_args.yaml"
vasp_kwargs = yaml.safe_load(open(params_file, 'r'))
vasp_kwargs['kpts'] = (3,2,1)
calc = Vasp(**vasp_kwargs)
neb = AIDNEB(start = images[0], 
            end = images[-1],
            interpolation = images[1:-1],
            calculator = copy.deepcopy(calc))
neb.run()