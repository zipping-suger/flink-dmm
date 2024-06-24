from load import load_composition, load_memory, load_space, load_robot

import json

import numpy as np
import torch

with open("task_sols.json") as f:
    task_sols = json.load(f)
    

X = []
Y = []

for task_sol in task_sols["task_sols"]:
    task_sol["composition"]
    init_box_pos = load_space(task_sol["space"])
    robot_state = load_robot(task_sol["robot"])
    end_box_pos = load_composition(task_sol["composition"])
    action = load_memory(task_sol["memories"]["memories"][0])
    
    X.append([*init_box_pos, *robot_state, *end_box_pos])
    Y.append(action)

X = torch.tensor(X)
Y = torch.tensor(Y)

print(X.shape)
print(Y.shape)

with open("X.trc", "wb") as f:
    torch.save(X, f)
with open("Y.trc", "wb") as f:
    torch.save(Y, f)


