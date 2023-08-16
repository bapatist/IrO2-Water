from ase.io import read
from ase.constraints import FixAtoms
from ase.optimize import BFGS
import yaml
from ase.calculators.vasp import Vasp

initial=read('left.xyz')
final=read('right.xyz')

constraint_l=FixAtoms(mask=[atom.tag > 0 for atom in initial])
constraint_r=FixAtoms(mask=[atom.tag > 0 for atom in final])

initial.set_constraint(constraint_l)
final.set_constraint(constraint_r)

# Set up VASP calculator for optimization and NEB
params_file = "vasp_args.yaml"
vasp_kwargs = yaml.safe_load(open(params_file, 'r'))
vasp_kwargs['kpts'] = (3,2,1)
calculator = Vasp(**vasp_kwargs)

# Optimize the initial and final images using VASP calculator
initial.set_calculator(calculator)
final.set_calculator(calculator)

optimizer = BFGS(initial, trajectory='initial_opt.traj')
optimizer.run(fmax=0.05)  # Adjust the force convergence criterion

optimizer = BFGS(final, trajectory='final_opt.traj')
optimizer.run(fmax=0.05)  # Adjust the force convergence criterion