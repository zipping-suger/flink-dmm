from onnxruntime import InferenceSession
import torch
from utils import *
import numpy as np
import json
import copy
import time

filename = "logs/2024_06_24_17_18_37/res.onnx"

sess = InferenceSession(filename)
dataset = LSTMJointDataset("X.trc", "Y.trc", "cpu")

with open("task_sols.json") as f:
    task_sols = json.load(f)

write_task_sols = {"task_sols": []}

inds = []

for i in range(len(task_sols["task_sols"])):
    input_data = dataset[i][0].unsqueeze(0).numpy()
    h0 = torch.zeros(*sess.get_inputs()[1].shape).numpy()
    c0 = torch.zeros(*sess.get_inputs()[2].shape).numpy()

    inputs = {
        sess.get_inputs()[0].name: input_data,
        sess.get_inputs()[1].name: h0,
        sess.get_inputs()[2].name: c0,
    }
    
    res = sess.run(None, inputs)
    joint_angles = torch.tensor(res[0][0]).view(-1)

    cur_task_sol = task_sols["task_sols"][i]

    write_task_sols["task_sols"].append(cur_task_sol)
    write_task_sols["task_sols"][-1]["memories"]["memories"].append(copy.deepcopy(cur_task_sol["memories"]["memories"][0]))
    write_task_sols["task_sols"][-1]["memories"]["memories"][1][0] = 1
    write_task_sols["task_sols"][-1]["memories"]["memories"][1][1]["pn"] = joint_angles.numpy().tolist()
    
    write_task_sols["task_sols"][-1]["memories"]["labels"].append(copy.deepcopy(cur_task_sol["memories"]["labels"][0]))
    write_task_sols["task_sols"][-1]["memories"]["labels"][1][0] = 1

    write_task_sols["task_sols"][-1]["memories"]["props"].append(copy.deepcopy(cur_task_sol["memories"]["props"][0]))
    write_task_sols["task_sols"][-1]["memories"]["props"][1][0] = 1
    write_task_sols["task_sols"][-1]["memories"]["props"][1][1]["created"] = int(time.time())

    inds.append((np.linalg.norm((joint_angles - dataset[i][1].view(-1))).item(), i))

inds.sort(reverse=True)

write_task_sols_sorted = {"task_sols": []}
for _, i in inds:
    write_task_sols_sorted["task_sols"].append(write_task_sols["task_sols"][i])

with open("task_sols_res.json", "w") as f:
    json.dump(write_task_sols_sorted, f)