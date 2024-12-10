import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import shutil
import datetime

import onnx_save
from utils import *
from models.cgan import VanillaDiscriminator, UnetGenerator, compute_discriminator_loss, compute_generator_loss
    
    
def train(dataloader, generator, discriminator, optimizer_g, optimizer_d, scheduler_g, scheduler_d, num_epochs, writer):
    # Training loop following the algorithm in the image
    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move data to device
            angles = targets.to(device)
            position = inputs.to(device)
            
            # Step 1: Update discriminator
            for _ in range(5):  # Crucial hyperparameter, need to balance between generator and discriminator
                # Draw noise samples
                z = torch.randn(position.size(0), generator.latent_size).to(device)
                
                # Generate fake samples
                fake_angles = generator(position, z)
                
                # Update discriminator
                d_loss, real_score, fake_score = compute_discriminator_loss(
                    discriminator, angles, fake_angles.detach(), position, device
                )
                
                optimizer_d.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1)
                optimizer_d.step()

            # Step 2 & 3: Update generator
            z = torch.randn(position.size(0), generator.latent_size).to(device)
            fake_angles = generator(position, z)
            g_loss = compute_generator_loss(discriminator, fake_angles, position)
            
            optimizer_g.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1)
            optimizer_g.step()

            # Log losses and scores
            if batch_idx % 100 == 0:
                writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch * len(dataloader) + batch_idx)
                writer.add_scalar('Loss/Generator', g_loss.item(), epoch * len(dataloader) + batch_idx)
                writer.add_scalar('Score/Real', real_score.item(), epoch * len(dataloader) + batch_idx)
                writer.add_scalar('Score/Fake', fake_score.item(), epoch * len(dataloader) + batch_idx)
            # # Debugging: Print gradients
            # for name, param in generator.named_parameters():
            #     if param.grad is not None:
            #         print(f"Generator Grad {name}: {param.grad.norm().item()}")
            # for name, param in discriminator.named_parameters():
            #     if param.grad is not None:
            #         print(f"Discriminator Grad {name}: {param.grad.norm().item()}")
        
        scheduler_g.step()
        scheduler_d.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Discriminator Loss: {d_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}")
        # # Save model checkpoints
        # if epoch % opt.save_interval == 0:
        #     onnx_save.export_cgan(generator, os.path.join(folder, f"cgan_{epoch}.onnx"), input_size)
            
        #     # set all models to eval mode and to device
        #     generator.to(device)
        #     generator.train()
        
        # onnx_save.export_cgan every 50 epochs
        if epoch % 50 == 49:
            onnx_save.export_cgan(generator, os.path.join(folder, f"cgan_{epoch}.onnx"), condition_size)
            generator.to(device)
            generator.train()
            discriminator.to(device)
            discriminator.train()
            



if __name__ == "__main__":
    global folder
    
     # Create output folder
    folder = os.path.join("logs", f"cgan_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")
    if os.path.exists(folder):
        shutil.rmtree(folder)
        
    parser = argparse.ArgumentParser(description='Train CGAN model.')
    args = parser.parse_args()
    
    # Hyperparameters
    num_epochs = 200
    batch_size = 128
    learning_rate_g = 1e-4
    learning_rate_d = 2e-4  # Reduced learning rate for discriminator

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset = JointDataset("X_filtered_mixed.trc", "Y_filtered_mixed.trc", device) # Use all the data
    dataset = JointDataset("X_filtered_mixed.trc", "Y_filtered_mixed.trc", device, allow_sub_sample=True, sub_sample_size=10000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    condition_size = dataset.input_size
    output_size = dataset.output_size
    
    # Initialize models
    generator = UnetGenerator(condition_size).to(device)
    discriminator = VanillaDiscriminator(condition_size, output_size).to(device)
    
    

    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate_d, betas=(0.5, 0.999))
    
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=50, gamma=0.9)
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=50, gamma=0.9)

    # Initialize SummaryWriter
    writer = SummaryWriter(folder)
    train(dataloader, generator, discriminator, optimizer_g, optimizer_d, scheduler_g, scheduler_d, num_epochs, writer)
    # Save the final model
    onnx_save.export_cgan(generator, os.path.join(folder, f"cgan.onnx"), condition_size)
    
    # Save models
    writer.close()