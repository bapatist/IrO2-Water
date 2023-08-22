import subprocess, os
import time
from pathlib import Path

def main():
    current_path = Path(os.getcwd())
    sim_indices  = range(int(input("sim_i start: ")),
                         1+int(input("sim_i end: ")),
                         1)
    run_eq       = int(input("Run sim_i/eq ? Enter 1 for yes and 0 for no:"))
    sub_sim_ind  = range(int(input("subfolder start: ")),
                         1+int(input("subfolder end: ")),
                         1)
    do_it(current_path, sim_indices, sub_sim_ind, run_eq)

def do_it(current_path, sim_indices, sub_sim_ind, run_eq):
    for i in sim_indices:
        if run_eq:
            eq_path = current_path/f"sim_{i}"/"eq"
            with open(eq_path/"run.sh", 'r') as file:
                lines = file.readlines()
            edited_lines = [line for line in lines if "dependency" not in line]
            with open(eq_path/"run.sh", 'w') as file:
                file.writelines(edited_lines)
            submit_eq_cmd = subprocess.run(["sbatch", "run.sh"],
                                           cwd = eq_path,
                                           capture_output=True, text=True)
            print(submit_eq_cmd.stdout)
            job_id = (submit_eq_cmd.stdout).split()[-1]
        for where_am_i, j in enumerate(sub_sim_ind):
            sub_sim_path = current_path/f"sim_{i}"/f"{j}"
            with open(sub_sim_path/"run.sh", 'r') as file:
                lines = file.readlines()
            if run_eq or where_am_i>0:
                for k, line in enumerate(lines):
                    if "dependency" in line:
                        index = line.find("afterok:")
                        lines[k] = line[:index + len("afterok:")] + job_id + '\n'
                with open(sub_sim_path/"run.sh", 'w') as file:
                    file.writelines(lines)
            else:
                edited_lines = [line for line in lines if "dependency" not in line]
                with open(sub_sim_path/"run.sh", 'w') as file:
                    file.writelines(edited_lines)

            submit_cmd = subprocess.run(["sbatch", "run.sh"],
                                        cwd = sub_sim_path,
                                        capture_output=True, text=True)
            print(submit_cmd.stdout)
            job_id = (submit_cmd.stdout).split()[-1]

if __name__=='__main__':
    main()
