import torch
import torch.nn as nn

# Assuming you have already defined and trained your model and saved it to "path_to_model"

# Load the model from the saved file
model = torch.load('./MACE_model.model')

# Move the model to the CPU if it was previously on GPU
model_cpu = model.to("cpu")

# Save the CPU model to a new file
path_to_model_cpu = "./CPU_MACE_model.model"  # Provide the desired path and filename for the CPU model
torch.save(model_cpu, path_to_model_cpu)
