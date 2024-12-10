import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

torch.set_default_dtype(torch.float32)

import os
import datetime

import onnx_save
from utils import *
from models.cvae import CVAE, vae_loss_function
  
torch.manual_seed(42)
folder = os.path.join("logs", f"cvae_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")

device = "cuda"

dataset = JointDataset("X_filtered.trc", "Y_filtered.trc", device)
cvae = CVAE(dataset.output_size, dataset.output_size, dataset.Y_normalise_params, dataset.Y_normalise_params, 8, dataset.input_size).to(device)

print(len(dataset))
n_epochs = 50 

optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-4)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)

writer = SummaryWriter(folder)

kl_annealing_start = 0  # Epoch to start annealing
kl_annealing_end = 20  # Epoch to reach full KL weight
max_kl_weight = 1.0    # Maximum weight for KL divergence


for epoch in range(0, n_epochs):
    print(f'Starting epoch {epoch+1}')
    # kl_weight = get_kl_weight(epoch, kl_annealing_start, kl_annealing_end, max_kl_weight)
    kl_weight  = 1.0
    print(f"KL Weight: {kl_weight:.4f}")
    train_loss = 0.0
    cvae.train()
    for i, data in enumerate(train_loader):
        inputs, targets = data

        optimizer.zero_grad()
        outputs, mu, logvar = cvae(targets, inputs)
        loss = vae_loss_function(outputs, targets, mu, logvar, kl_weight)
        
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
    construct_loss = 0.0
    sampled_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            inputs, targets = data

            outputs, mu, logvar = cvae(targets, inputs) # x, c
            loss = vae_loss_function(outputs, targets, mu, logvar, kl_weight)
            
            c_loss = torch.nn.functional.mse_loss(outputs, targets, reduction='mean')
            
            sampled_outputs = cvae.sample(inputs)
            s_loss = torch.nn.functional.mse_loss(sampled_outputs, targets)
            
            if torch.isnan(loss):
                print(f"NaN detected in validation loss at epoch {epoch}, batch {i}")
                
                
            valid_loss += loss.item()
            construct_loss += c_loss.item()
            sampled_loss += s_loss.item()
    print("valid", epoch, valid_loss / len(valid_loader))
    print("construct", epoch, construct_loss / len(valid_loader))
    print("sampled", epoch, sampled_loss / len(valid_loader))
    
    writer.add_scalar("Loss/valid", valid_loss / len(valid_loader), epoch)
    writer.add_scalar("Loss/construct", construct_loss / len(valid_loader), epoch)
    
    # Onnx save the model every 10 epochs
    if epoch % 10 == 9:
        onnx_save.export_cvae(cvae, os.path.join(folder, f"cvae_epoch_{epoch+1}.onnx"), dataset.input_size)
        # Model to cuda
        cvae.to(device)
        cvae.train()
        


# Save the final model
onnx_save.export_cvae(cvae, os.path.join(folder, f"cvae.onnx"), dataset.input_size)    
torch.save(cvae.state_dict(), os.path.join(folder, f"cvae.pt"))