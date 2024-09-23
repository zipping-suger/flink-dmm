import json

import numpy as np
import torch

with open("single_task_sols_0.9.json") as f:
    task_sols = json.load(f)
    

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


