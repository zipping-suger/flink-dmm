from onnxruntime import InferenceSession
import torch
from utils import *
import numpy as np
import json
import copy
import time

filename = "logs/2024_06_21_17_30_13/res.onnx"

sess = InferenceSession(filename)
dataset = JointDataset("X.trc", "Y.trc", "cpu")



with open("task_sols.json") as f:
    task_sols = json.load(f)

write_task_sols = {"task_sols": []}

inds = []

for i in range(len(task_sols["task_sols"])):
    res = torch.tensor(sess.run(None, {sess.get_inputs()[0].name: dataset[i][0].unsqueeze(0).numpy()})[0][0])
    cur_task_sol = task_sols["task_sols"][i]

    write_task_sols["task_sols"].append(cur_task_sol)
    write_task_sols["task_sols"][-1]["memories"]["memories"].append(copy.deepcopy(cur_task_sol["memories"]["memories"][0]))
    write_task_sols["task_sols"][-1]["memories"]["memories"][1][0] = 1
    write_task_sols["task_sols"][-1]["memories"]["memories"][1][1]["pn"] = res.numpy().tolist()
    
    write_task_sols["task_sols"][-1]["memories"]["labels"].append(copy.deepcopy(cur_task_sol["memories"]["labels"][0]))
    write_task_sols["task_sols"][-1]["memories"]["labels"][1][0] = 1

    write_task_sols["task_sols"][-1]["memories"]["props"].append(copy.deepcopy(cur_task_sol["memories"]["props"][0]))
    write_task_sols["task_sols"][-1]["memories"]["props"][1][0] = 1
    write_task_sols["task_sols"][-1]["memories"]["props"][1][1]["created"] = int(time.time())

    inds.append((np.linalg.norm((res - dataset[i][1])).item(), i))

inds.sort(reverse=True)

write_task_sols_sorted = {"task_sols": []}
for _, i in inds:
    write_task_sols_sorted["task_sols"].append(write_task_sols["task_sols"][i])

with open("task_sols_res.json", "w") as f:
    json.dump(write_task_sols_sorted, f)