job_type = "A_H2O"

sim_indexes = [0, 3]

import glob

from pathlib import Path
import os
for i in range(sim_indexes[0], 1+sim_indexes[1], 1):
    folder = f"sim_{i}"
    cwd                 = Path(os.getcwd())
    path_to_sim = cwd/folder
    for j in range(2,10,1):
        with open("./run.sh", 'rt') as runner:
            data_run = runner.read()
            data_run = data_run.replace('JOB_NAME', f'{job_type}_{i}_{j}')
        with open(path_to_sim/f"{j}/run.sh", 'wt') as new_runner:
            new_runner.write(data_run)

