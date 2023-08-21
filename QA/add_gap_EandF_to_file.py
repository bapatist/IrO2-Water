#%%
from ase.io import read, write
from quippy.potential import Potential
#%%
def set_gap_energy(gap_file, inp_struc_file, out_struc_file):
    pot = Potential(param_filename=gap_file)
    struc_set = read(inp_struc_file, ":")
    len(struc_set)
    for im in struc_set:
        im.set_calculator(pot)
        im.info['GAP_energy'] = im.get_potential_energy()
        im.arrays['GAP_forces'] = im.get_forces()
    write(out_struc_file, struc_set)
#%%
set_gap_energy("GAPs/I4/lowF/GAP/GAP_0.xml", 
               inp_struc_file="./TRAIN_SET/I4/low_F_TRAINING_ITER4_20_WITHOUTGAP.xyz",
               out_struc_file="./TRAIN_SET/I4/ts_lowF.xyz")
# %%
set_gap_energy("GAPs/I4/lowF/GAP/GAP_0.xml", 
               inp_struc_file="TEST_SET/I4/test_ITER4.xyz",  
               out_struc_file="TEST_SET/I4/test_lowF.xyz")
# %%
set_gap_energy("GAPs/I4/all/GAP_21.xml",
               inp_struc_file="./TRAIN_SET/I4/low_F_TRAINING_ITER4_20_WITHOUTGAP.xyz",
               out_struc_file="./TRAIN_SET/I4/TRAINING_ITER4_21_WITHOUTGAP.xyz")
# %%
set_gap_energy("GAPs/I4/GAP_20.xml", 
               inp_struc_file="TEST_SET/test_set_IF.xyz",  
               out_struc_file="TEST_SET/test_IF_all_FFS.xyz")
# %%
