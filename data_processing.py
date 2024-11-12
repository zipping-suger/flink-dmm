import os
import warnings
import torch
from utils import load_data_from_json, filter_data_dist


raw_task_sols_folder = "raw_task_sols_l_r_1104"
for filename in os.listdir(raw_task_sols_folder):
    
    if filename.startswith("single_task_sols_") and filename.endswith(".json"):
        X, Y = load_data_from_json(os.path.join(raw_task_sols_folder, filename))
        r = float(filename.split("_")[-1].replace(".json", ""))
        
        print(f"Processing file: {filename}")
        print(f"Filtering data with r = {r}")
        
        X_filtered, Y_filtered = filter_data_dist(X, Y, r)
        
        print(f"Filtered data shapes: {X_filtered.shape}, {Y_filtered.shape}")
        print("------------------------------------------------")
        
        os.makedirs("tmp", exist_ok=True)
        
        with open(f"tmp/X_filtered_{r}.trc", "wb") as f:
            torch.save(torch.tensor(X_filtered), f)
        with open(f"tmp/Y_filtered_{r}.trc", "wb") as f:
            torch.save(torch.tensor(Y_filtered), f)
            
            
# Concatenate all the filtered data
warnings.filterwarnings("ignore")
all_X = []
all_Y = []

for filename in os.listdir("tmp"):
    if filename.startswith("X_filtered_") and filename.endswith(".trc"):
        X = torch.load(os.path.join("tmp", filename))
        all_X.append(X)
    elif filename.startswith("Y_filtered_") and filename.endswith(".trc"):
        Y = torch.load(os.path.join("tmp", filename))
        all_Y.append(Y)

all_X = torch.cat(all_X, dim=0)
all_Y = torch.cat(all_Y, dim=0)


# save as .trc files
with open("X_filtered.trc", "wb") as f:
    torch.save(all_X, f)
with open("Y_filtered.trc", "wb") as f:
    torch.save(all_Y, f)
    
# # clear tmp folder
# for filename in os.listdir("tmp"):
#     os.remove(os.path.join("tmp", filename))
    
print(f"Final data shapes: {all_X.shape}, {all_Y.shape}")


        


