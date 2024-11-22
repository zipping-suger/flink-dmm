import argparse
import os
import numpy as np
import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import JointDataset

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=0, help="dimensionality of the latent space")
parser.add_argument("--save_interval", type=int, default=50, help="interval between image sampling")
parser.add_argument("--input_emb_size", type=int, default=18, help="size of the input embedding")
opt = parser.parse_args()
print(opt)

class CGANSampleWrapper(nn.Module):
    def __init__(self, generator):
        super(CGANSampleWrapper, self).__init__()
        self.generator = generator
    def forward(self, inputs):
        input_emb = inputs
        # noise to cpu
        noise = torch.randn(inputs.size(0), opt.latent_dim).to("cpu")
        gen_trajs = self.generator(input_emb, noise)
        return gen_trajs
    
def export_cgan(generator: torch.nn.Module,  output_name: str, input_size: int):
    
    generator = generator.to("cpu")
    generator.eval()
    
    # Create dummy inputs
    dummy_input = torch.randn(1, input_size)
    dynamic_axes = {'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}}

    # Initialize the wrapper model
    wrapper_model = CGANSampleWrapper(generator)
    wrapper_model.to("cpu")
    wrapper_model.eval()

    # Export the wrapper model
    torch.onnx.export(
        wrapper_model, 
        dummy_input,  # Model input
        output_name,  # Output file name
        input_names=['input'],  # Input names
        output_names=['output'],  # Output names
        dynamic_axes=dynamic_axes  # Variable length axes
    )

class Generator(nn.Module):
    def __init__(self, emb_size, output_size, latent_dim):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + emb_size, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, output_size),
            nn.Tanh()
        )

    def forward(self, input_emb, noise):
        gen_input = torch.cat((input_emb, noise), -1)
        return self.model(gen_input)

class Discriminator(nn.Module):
    def __init__(self, emb_size, output_size):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(emb_size + output_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()  # Added sigmoid to output probabilities
        )

    def forward(self, input_emb, traj):
        d_in = torch.cat((input_emb, traj), -1)
        return self.model(d_in)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = JointDataset("X_filtered_mixed.trc", "Y_filtered_mixed.trc", device)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

input_size = dataset.input_size
output_size = dataset.output_size

# Initialize generator and discriminator
generator = Generator(opt.input_emb_size, output_size, opt.latent_dim).to(device)
discriminator = Discriminator(opt.input_emb_size, output_size).to(device)

# Binary cross entropy loss
criterion = nn.BCELoss()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Create output folder
folder = os.path.join("logs", datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
os.makedirs(folder, exist_ok=True)
writer = SummaryWriter(folder)

def train_discriminator(real_data, input_emb, noise):
    batch_size = real_data.size(0)
    
    # Reset gradients
    optimizer_D.zero_grad()
    
    # Train on real data
    prediction_real = discriminator(input_emb, real_data)
    # Calculate error and backpropagate
    error_real = criterion(prediction_real, torch.ones(batch_size, 1).to(device))

    # Generate fake data
    fake_data = generator(input_emb, noise)
    # Train on fake data
    prediction_fake = discriminator(input_emb, fake_data.detach())
    # Calculate error and backpropagate
    error_fake = criterion(prediction_fake, torch.zeros(batch_size, 1).to(device))
    
    error = (error_real + error_fake)/2
    error.backward()
    
    # Update weights with gradients
    optimizer_D.step()
    
    return error

def train_generator(input_emb, noise):
    # Reset gradients
    optimizer_G.zero_grad()
    
    # Generate fake data
    fake_data = generator(input_emb, noise)
    # Get discriminator's prediction
    prediction = discriminator(input_emb, fake_data)
    # Calculate error and backpropagate
    error = criterion(prediction, torch.ones(noise.size(0), 1).to(device))
    error.backward()
    
    # Update weights with gradients
    optimizer_G.step()
    
    return error

# Training loop
k_steps = 5  # Number of steps to train the discriminator
for epoch in range(opt.n_epochs):
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Move data to device
        real_data = targets.to(device)
        input_emb = inputs.to(device)
        
        # Train Discriminator
        noise = torch.randn(inputs.size(0), opt.latent_dim).to(device)
        d_loss = train_discriminator(real_data, input_emb, noise)
        
        # Train Generator
        for _ in range(k_steps):
            noise = torch.randn(inputs.size(0), opt.latent_dim).to(device)
            g_loss = train_generator(input_emb, noise)
            
        # Log progress
        if batch_idx % 100 == 0:
            print(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {batch_idx}/{len(dataloader)}] "
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
            
            # Log to TensorBoard
            writer.add_scalar('D_loss', d_loss.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar('G_loss', g_loss.item(), epoch * len(dataloader) + batch_idx)
            
    # Save model checkpoints
    if epoch % opt.save_interval == 0:
        export_cgan(generator, os.path.join(folder, f"cgan_{epoch}.onnx"), input_size)
        
        # set all models to eval mode and to device
        generator.to(device)
        generator.train()

# Save the final model
export_cgan(generator, os.path.join(folder, f"cgan_{epoch}.onnx"), input_size)
    