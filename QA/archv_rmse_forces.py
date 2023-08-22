
#%%
import numpy as np
from ase.io import read
from wfl.plotting.plot_ef_correlation import extract_energies_per_atom, plot_energy, plot, extract_forces, plot_force_components, rms_text_for_plots
from matplotlib import pyplot as plt
import matplotlib
font = {'size': 18}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
# %% FORCE ERROR AND SCATTER
train_Ir, train_O, train_H = [], [], []
test_Ir,  test_O,  test_H  = [], [], []
for config in train_set:
    config.arrays['GAP_forces'] = config.get_forces()
    train_Ir.append(config[[atom.index for atom in config if atom.symbol=='Ir']])
    train_O.append(config[[atom.index for atom in config if  atom.symbol=='O']])
    train_H.append(config[[atom.index for atom in config if  atom.symbol=='H']])
for config in test_set:
    config.arrays['GAP_forces'] = config.get_forces()
    test_Ir.append(config[[atom.index for atom in config if atom.symbol=='Ir']])
    test_O.append(config[[atom.index for atom in config if atom.symbol=='O']])
    test_H.append(config[[atom.index for atom in config if atom.symbol=='H']])

gap_train_Ir, gap_train_O, gap_train_H = (extract_forces(train_Ir, "GAP_forces", flat=True)['Ir'],      
                                          extract_forces(train_O,  "GAP_forces", flat=True)['O'],      
                                          extract_forces(train_H,  "GAP_forces", flat=True)['H'])
gap_test_Ir,  gap_test_O,  gap_test_H  = (extract_forces(test_Ir,  "GAP_forces", flat=True)['Ir'],      
                                          extract_forces(test_O,   "GAP_forces", flat=True)['O'],      
                                          extract_forces(test_H,   "GAP_forces", flat=True)['H'])

dft_train_Ir, dft_train_O, dft_train_H =  (extract_forces(train_Ir, "DFT_forces", flat=True)['Ir'],      
                                           extract_forces(train_O, "DFT_forces", flat=True)['O'],      
                                           extract_forces(train_H, "DFT_forces", flat=True)['H'])
dft_test_Ir,  dft_test_O, dft_test_H   =  (extract_forces(test_Ir,  "DFT_forces", flat=True)['Ir'],      
                                           extract_forces(test_O,  "DFT_forces", flat=True)['O'],      
                                           extract_forces(test_H,  "DFT_forces", flat=True)['H'])
#%% FORCE RMSEs
err_txt_f_train_ir = rms_text_for_plots(gap_train_Ir, dft_train_Ir)[5:]
err_txt_f_train_o  = rms_text_for_plots(gap_train_O,  dft_train_O)[5:]
err_txt_f_train_h  = rms_text_for_plots(gap_train_H,  dft_train_H)[5:]

err_f_train_ir = float(err_txt_f_train_ir.split()[0])
err_f_train_o  = float(err_txt_f_train_o.split()[0])
err_f_train_h  = float(err_txt_f_train_h.split()[0])
print("For train set having", f"{len(train_set)} structures \n",
        f"RMSE forces on Ir= {err_txt_f_train_ir}\n\n",
        f"RMSE forces on O = {err_txt_f_train_o}\n\n",
        f"RMSE forces on H = {err_txt_f_train_h}\n\n",)

err_txt_f_test_ir = rms_text_for_plots(gap_test_Ir, dft_test_Ir)[5:]
err_txt_f_test_o  = rms_text_for_plots(gap_test_O,  dft_test_O)[5:]
err_txt_f_test_h  = rms_text_for_plots(gap_test_H,  dft_test_H)[5:]

err_f_test_ir = float(err_txt_f_test_ir.split()[0])
err_f_test_o  = float(err_txt_f_test_o.split()[0])
err_f_test_h  = float(err_txt_f_test_h.split()[0])
print("For test set having", f"{len(test_set)} structures \n",
        f"RMSE forces on Ir= {err_txt_f_test_ir}\n\n",
        f"RMSE forces on O = {err_txt_f_test_o}\n\n",
        f"RMSE forces on H = {err_txt_f_test_h}\n\n",)


no_of_Ir_atoms = len([atom for atom in train_set[0] if atom.symbol=='Ir'])
no_of_O_atoms =  len([atom for atom in train_set[0] if atom.symbol=='O'])
no_of_H_atoms =  len([atom for atom in train_set[0] if atom.symbol=='H'])
# print(no_of_Ir_atoms, no_of_O_atoms, no_of_H_atoms)
avg_rmse_force_train =  (no_of_Ir_atoms*err_f_train_ir + 
                         no_of_O_atoms*err_f_train_o + 
                         no_of_H_atoms*err_f_train_h)/len((train_set[0]))
avg_rmse_force_test =  (no_of_Ir_atoms*err_f_test_ir + 
                         no_of_O_atoms*err_f_test_o + 
                         no_of_H_atoms*err_f_test_h)/len((train_set[0]))


print(f"Average RMSE force train {avg_rmse_force_train}")
print(f"Average RMSE force test  {avg_rmse_force_test}")

# %%
