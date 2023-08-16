from ase.io import read, write
from ase.constraints import FixAtoms
from ase.neb import NEB
from ase.optimize import FIRE
from ase.io.trajectory import Trajectory
from ase.calculators.vasp import Vasp
import yaml

initial=read('left_opt.xyz')
final=read('right_opt.xyz')

constraint=FixAtoms(mask=[atom.tag > 0 for atom in initial])

Nimages=5

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
for i in range(Nimages+2):
    write("images"+str(i+1)+".png",images[i])

params_file = "vasp_args.yaml"
vasp_kwargs = yaml.safe_load(open(params_file, 'r'))
vasp_kwargs['kpts'] = (3,2,1)

for image in images[1:Nimages+1]:
    calc = Vasp(**vasp_kwargs)
    image.set_calculator(calc)

dyn=FIRE(neb, trajectory='A2B.traj', logfile='neb.log')
for i in range(1,Nimages+1):
    traj = Trajectory('neb-%d.traj' % i, 'w', images[i])
    dyn.attach(traj)

dyn.run(fmax=0.05)
