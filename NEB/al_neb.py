from ase import Atoms
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.neb import NEB
from ase.optimize import FIRE
from ase.io.trajectory import Trajectory
from ase.calculators.vasp import Vasp
import quippy
import copy, os
import yaml
from dataclasses import dataclass
from pathlib import Path
from wfl.fn_iterWfl import run_dft, set_force_sigma, get_gap_multistage
os.environ['WFL_NUM_PYTHON_SUBPROCESSES'] = "16"
os.environ['WFL_GAP_FIT_OMP_NUM_THREADS'] = "32"

@dataclass
class ActiveLearning():
    left:    Atoms  #expected to have tag=1 for fixed atoms
    right:   Atoms  #expected to have tag=1 for fixed atoms
    nimages: int
    vasp_kwargs: dict

    def __post_init__(self) -> None:
        self.vasp_calc = Vasp(**self.vasp_kwargs)

    def minimize_left_right(self, fmax=0.05):
        initial, final = self.left, self.right
        constraint_l=FixAtoms(mask=[atom.tag > 0 for atom in initial])
        constraint_r=FixAtoms(mask=[atom.tag > 0 for atom in final])
        
        initial.set_constraint(constraint_l)
        final.set_constraint(constraint_r)

        calculator = self.vasp_calc
        initial.set_calculator(calculator)
        final.set_calculator(calculator)

        optimizer = FIRE(initial, trajectory='left_opt.traj')
        optimizer.run(fmax=fmax)  # Adjust the force convergence criterion

        optimizer = FIRE(final, trajectory='right_opt.traj')
        optimizer.run(fmax=fmax)  # Adjust the force convergence criterion
        
        self.left  = Trajectory('left_opt.traj')[-1]
        self.right = Trajectory('right_opt.traj')[-1]
        write("left_opt.xyz",  self.left)
        write("right_opt.xyz", self.right)
    
    def interpolate_idpp(self):
        constraint=FixAtoms(mask=[atom.tag > 0 for atom in self.left])
        guess_path=[self.left]
        for i in range(self.nimages):
                image=self.left.copy()
                image.set_constraint(constraint)
                guess_path.append(image)
        guess_path.append(self.right)
        neb=NEB(guess_path,climb=True)
        neb.interpolate('idpp', apply_constraint=True)
    
        write("guess_path.xyz", guess_path)
        return guess_path
    
    def run_neb_gap(self, iter, guess_path, path_to_xml, fmax=0.05):
        constraint=FixAtoms(mask=[atom.tag > 0 for atom in self.left])
        for im in guess_path:
            im.set_constraint(constraint)
        
        neb=NEB(guess_path,climb=True)
        
        calc_base = quippy.potential.Potential(param_filename=path_to_xml)
        for image in guess_path[1:self.nimages+1]:
            calc = copy.deepcopy(calc_base)
            image.set_calculator(calc)

        dyn=FIRE(neb, trajectory=f'A2B_{iter}.traj', logfile=f'neb_{iter}.log')
        for i in range(1,self.nimages+1):
            traj = Trajectory('neb-%d.traj' % i, 'w', guess_path[i])
            dyn.attach(traj)
        dyn.run(fmax=fmax)
        mep = Trajectory(f"A2B_{iter}.traj")[-self.nimages:]
        return mep
    
    def run_dft_wfl(self, iteration, mep, params_file, kpoints, out_prefix="mep_inter_dft"):
        out_file = f'{out_prefix}_{iteration}.xyz'
        mep_without_ends = mep[1:-1]
        run_dft(in_file=mep_without_ends, 
                out_file=out_file, 
                params_file=params_file, 
                kpoints=kpoints)
        os.system(f"sed -i 's/DFT_stress=/DFT_stress_FOO=/g' {out_file}")
        dft_atoms_list = read(out_file, ':')
        for im in dft_atoms_list:
            set_force_sigma(im, proportion=0.01) # Setting Per Atom Force Sigma
            im.info['config_type'] = f"MEP_sample_{iteration}"
        try:
            im.info['DFT_energy']
        except KeyError:
            dft_atoms_list.remove(im)
            print(f" WARNING: A DFT calc was failed and removed from samples: iteration {iteration}")
        return dft_atoms_list
    
def main():
    left, right = read("./left_opt.xyz"), read("./right_opt.xyz")
    params_file = "vasp_args.yaml"
    Zs = [77, 8, 1]
    length_scales = yaml.safe_load(open('length_scales.yaml', 'r'))
    ms_params_file = 'multistage_settings_new.yaml'

    vasp_kwargs = yaml.safe_load(open(params_file, 'r'))
    vasp_kwargs['kpts'] = (3,2,1)

    AL = ActiveLearning(left=left, right=right, 
                        vasp_kwargs=vasp_kwargs,
                        nimages=8)
    # training    = "/ptmp/nbapat/surf/ITER4/GAP/training_21.xyz" #initial TS
    # path_to_xml = "/ptmp/nbapat/surf/ITER4/GAP/GAP_21.xml"      #initial GAP 
    training    = "./GAP/training_1.xyz" #initial TS
    path_to_xml = "./GAP/GAP_1.xml"      #initial GAP   
    # AL.minimize_left_right(fmax=0.05)
    for i in range(1,10,1):
        guess_path = AL.interpolate_idpp()
        mep        = AL.run_neb_gap(guess_path=guess_path,
                                    iter=i, 
                                    path_to_xml=path_to_xml,
                                    fmax=0.08)
        dft_mep    = AL.run_dft_wfl(iteration=i,
                                    mep=mep, 
                                    params_file=params_file, 
                                    kpoints=vasp_kwargs['kpts'])
        write(f'GAP/training_{i+1}.xyz', read(training, ':') + dft_mep)
        training    = f'GAP/training_{i+1}.xyz'
        get_gap_multistage(training, Zs, length_scales, ms_params_file, i+1)
        path_to_xml = f'GAP/GAP_{i+1}.xml'

if __name__ == "__main__":
    main()