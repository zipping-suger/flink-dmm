import json

import numpy as np
import torch
import os

task_sols = {"task_sols": []}

raw_task_sols_folder = "raw_task_sols_l_r_test"
for filename in os.listdir(raw_task_sols_folder):
    if filename.startswith("single_task_sols_") and filename.endswith(".json"):
        with open(os.path.join(raw_task_sols_folder, filename)) as f:
            data = json.load(f)
            task_sols["task_sols"].extend(data["task_sols"])
    

X = []
Y = []

for task_sol in task_sols["task_sols"]:
    X.append(task_sol["condensed_task"])
    Y.append(task_sol["solution"])
    
X = torch.tensor(X)
Y = torch.tensor(Y)

print(X.shape)
print(Y.shape)

with open("X.trc", "wb") as f:
    torch.save(X, f)
with open("Y.trc", "wb") as f:
    torch.save(Y, f)


