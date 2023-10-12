#%%
from ase.io import read, write
#%%
ts = read("./TRAIN_SET/TRAINING_ITER4_20_WITHOUTGAP.xyz", ":")
len(ts)
#%%
low_F_strucs = []
high_F_strucs = []
for im in ts:
    stop =  False
    dft_forces = im.arrays['DFT_forces']
    for f in dft_forces:
        if stop==False:   
            for f_c in f:
                if abs(f_c) > 10:
                    high_F_strucs.append(im)
                    print("Found a high F struc with config type: ", im.info['config_type'])
                    stop = True
                    break
        else:
            break
    if stop == False:
        low_F_strucs.append(im)
#%%
print(len(low_F_strucs), len(high_F_strucs))
# %%
# write("./TRAIN_SET/high_F_test_strucs.xyz", high_F_strucs)
write("./TRAIN_SET/low_F_TRAINING_ITER4_20_WITHOUTGAP.xyz", low_F_strucs)
# %%
from ase.visualize import view
view(high_F_strucs)
# %%
