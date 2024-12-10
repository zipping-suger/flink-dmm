import torch
import torch.nn as nn

class LSTM(nn.Module):
  def __init__(self, in_size, hidden_size, out_size, num_layers, in_norm, out_norm):
    super().__init__()
    self.lstm = nn.LSTM(in_size, hidden_size, num_layers, batch_first=True)
    self.lstm.num_layers
    self.layers = nn.Sequential(
        nn.Linear(hidden_size, 32),
        nn.ELU(),
        nn.Linear(32, 16),
        nn.ELU(),
        nn.Linear(16, out_size)
    )

    self.in_norm = in_norm
    self.out_norm = out_norm

  def forward(self, x, hidden):
    x = (x - self.in_norm[0]) / self.in_norm[1]
    x, hidden = self.lstm(x, hidden)
    x = self.layers(x)
    return x * self.out_norm[1] + self.out_norm[0], hidden

  def to(self, device):
    self.in_norm[0] = self.in_norm[0].to(device)
    self.in_norm[1] = self.in_norm[1].to(device)
    self.out_norm[0] = self.out_norm[0].to(device)
    self.out_norm[1] = self.out_norm[1].to(device)

    return super().to(device)