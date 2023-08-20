from ase.io import read
left  = read("./left_opt.xyz")
right = read("./right_opt.xyz")

E_diff = left.info['energy'] - right.info['energy']
print(E_diff)