import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

import os
import onnx_save
import datetime

from utils import *
from models.lstm import LSTM
  
torch.manual_seed(42)
folder = os.path.join("logs", datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

device = "cuda"
INPUT_SIZE=19
OUTPUT_SIZE=12
HIDDEN_SIZE=64
NUM_HIDDEN=1
BATCH_SIZE=32

dataset = LSTMJointDataset("dual_X.trc", "dual_Y.trc", device, OUTPUT_SIZE)

lstm = LSTM(INPUT_SIZE, 
            HIDDEN_SIZE,
            OUTPUT_SIZE, 
            NUM_HIDDEN,
            dataset.X_normalise_params, 
            dataset.Y_normalise_params).to(device)

n_epochs = 500

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-3)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
# scheduler = lr_scheduler.ConstantLR(optimizer, 1)

train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

writer = SummaryWriter(folder)

for epoch in range(0, n_epochs):
    print(f'Starting epoch {epoch+1}')
    train_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, targets = data

        optimizer.zero_grad()
        hidden = (torch.zeros(NUM_HIDDEN, inputs.shape[0], HIDDEN_SIZE).to(device),
                  torch.zeros(NUM_HIDDEN, inputs.shape[0], HIDDEN_SIZE).to(device))
        outputs, hidden = lstm(inputs, hidden)
        loss = loss_function(outputs, targets)
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

            hidden = (torch.zeros(NUM_HIDDEN, inputs.shape[0], HIDDEN_SIZE).to(device),
                      torch.zeros(NUM_HIDDEN, inputs.shape[0], HIDDEN_SIZE).to(device))
            outputs, hidden = lstm(inputs, hidden)
            loss = loss_function(outputs, targets)
            valid_loss += loss.item()
    print("valid", epoch, valid_loss / len(valid_loader))
    writer.add_scalar("Loss/valid", valid_loss / len(valid_loader), epoch)

onnx_save.export_lstm(lstm, os.path.join(folder, "res.onnx"), INPUT_SIZE, HIDDEN_SIZE, NUM_HIDDEN)
