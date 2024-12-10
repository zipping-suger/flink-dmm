import torch
import torch.nn as nn

def loss_function(outputs, targets):
    # Basic loss (e.g., Mean Squared Error)
    basic_loss = torch.nn.functional.mse_loss(outputs, targets)
    
    return basic_loss

class MLP(nn.Module):
  def __init__(self, in_size, out_size, in_norm, out_norm):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_size, 32),
        nn.ELU(),
        nn.Linear(32, 64),
        nn.ELU(),
        nn.Linear(64, 128),
        nn.ELU(),
        nn.Linear(128, 256),
        nn.ELU(),
        nn.Linear(256, 256),
        nn.ELU(),
        nn.Linear(256, out_size)
    )

    self.in_norm = in_norm
    self.in_norm[1][self.in_norm[1] == 0] = 1
    self.out_norm = out_norm
    self.out_norm[1][self.out_norm[1] == 0] = 1

  def forward(self, x):
    x = (x - self.in_norm[0]) / self.in_norm[1]
    x = self.layers(x)
    return x * self.out_norm[1] + self.out_norm[0]

  def to(self, device):
    self.in_norm[0] = self.in_norm[0].to(device)
    self.in_norm[1] = self.in_norm[1].to(device)
    self.out_norm[0] = self.out_norm[0].to(device)
    self.out_norm[1] = self.out_norm[1].to(device)

    return super().to(device)