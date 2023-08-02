import torch
torch.cuda.empty_cache()

import gc
del variables
gc.collect()
