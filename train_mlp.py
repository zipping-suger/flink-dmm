import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

torch.set_default_dtype(torch.float32)

import os
import onnx_save
from utils import *
import datetime
  
torch.manual_seed(42)
folder = os.path.join("logs", datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

device = "cuda"

dataset = JointDataset("X_filtered.trc", "Y_filtered.trc", device)
mlp = MLP(dataset.input_size, dataset.output_size, dataset.X_normalise_params, dataset.Y_normalise_params).to(device)
print(len(dataset))
n_epochs =50
    

optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
# scheduler = lr_scheduler.ConstantLR(optimizer, 1)

train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)

writer = SummaryWriter(folder)

for epoch in range(0, n_epochs):
    print(f'Starting epoch {epoch+1}')
    train_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, targets = data

        optimizer.zero_grad()
        outputs = mlp(inputs)
        loss = loss_function(outputs, targets)
        
        # check for NaNs
        if torch.isnan(loss):
            print(f"NaN detected in training loss at epoch {epoch}, batch {i}")
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print("train", epoch, train_loss / len(train_loader))
    writer.add_scalar("Loss/train", train_loss / len(train_loader), epoch)
    writer.add_scalar("Training/LR", scheduler.get_last_lr()[0], epoch)

    if epoch % 10 == 9:
        scheduler.step()

    valid_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            inputs, targets = data

            outputs = mlp(inputs)
            loss = loss_function(outputs, targets)
            
            if torch.isnan(loss):
                print(f"NaN detected in validation loss at epoch {epoch}, batch {i}")
                
                loss = loss_function(outputs, targets)
                
            valid_loss += loss.item()
    print("valid", epoch, valid_loss / len(valid_loader))
    writer.add_scalar("Loss/valid", valid_loss / len(valid_loader), epoch)

# onnx_save.export(mlp, os.path.join(folder, "res.onnx"), dataset.input_size)
onnx_save.export(mlp, os.path.join(folder, f"dmm_{datetime.datetime.now().strftime('%Y_%m_%d')}.onnx"), dataset.input_size)
