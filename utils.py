import torch
import torch.nn as nn
import torch.utils.data

# Loss function also penalizes the difference of velocity, acceleration, and initial/final state variation
def loss_function(outputs, targets, velocity_coeff=1.0, acceleration_coeff=1.0, jerk_coeff=1.0, state_coeff=1):
    # Basic loss (e.g., Mean Squared Error)
    basic_loss = torch.nn.functional.mse_loss(outputs, targets)
    
    # Calculate velocity (first derivative)
    velocity_outputs = torch.diff(outputs, dim=0)
    velocity_targets = torch.diff(targets, dim=0)
    
    # Calculate acceleration (second derivative)
    acceleration_outputs = torch.diff(velocity_outputs, dim=0)
    acceleration_targets = torch.diff(velocity_targets, dim=0)
    
    # Calculate jerk (third derivative)
    jerk_outputs = torch.diff(acceleration_outputs, dim=0)
    jerk_targets = torch.diff(acceleration_targets, dim=0)
    
    # Calculate velocity penalty
    velocity_penalty = torch.nn.functional.mse_loss(velocity_outputs, velocity_targets)
    
    # Calculate acceleration penalty
    acceleration_penalty = torch.nn.functional.mse_loss(acceleration_outputs, acceleration_targets)
    
    # Calculate jerk penalty
    jerk_penalty = torch.nn.functional.mse_loss(jerk_outputs, jerk_targets)
    
    # Calculate initial state penalty
    initial_state_penalty = torch.nn.functional.mse_loss(outputs[0], targets[0])
    
    # Calculate final state penalty
    final_state_penalty = torch.nn.functional.mse_loss(outputs[-1], targets[-1])
    
    # Calculate L2 norm of the jerk
    jerk_l2_norm = torch.norm(jerk_outputs, p=2)
    
    # Combine basic loss with penalties
    total_loss = (basic_loss 
                  + velocity_coeff * velocity_penalty 
                  + acceleration_coeff * acceleration_penalty 
                  + jerk_coeff * jerk_penalty
                  + state_coeff * initial_state_penalty
                  + state_coeff * final_state_penalty)
    
    return total_loss

class JointDataset(torch.utils.data.Dataset):
    def __init__(self, x_filename, y_filename, device):
        with open(x_filename, "rb") as f:
            self.X = torch.load(f).to(device)
        with open(y_filename, "rb") as f:
            self.Y = torch.load(f).to(device)

        assert (self.X.shape[0] == self.Y.shape[0])

        self.X_normalise_params = [self.X.mean(dim=0), self.X.std(dim=0)]
        self.Y_normalise_params = [self.Y.mean(dim=0), self.Y.std(dim=0)]

        self.rows = self.X.shape[0]
        self.input_size = self.X.shape[1]
        self.output_size = self.Y.shape[1]

    def __len__(self):
        return self.rows

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


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


class LSTMJointDataset(torch.utils.data.Dataset):
    TIME_LENGTH = 120

    def __init__(self, x_filename, y_filename, device, output):
        with open(x_filename, "rb") as f:
            self.X = torch.load(f).to(device)
            self.X = self.X[:, None, :].expand(-1, self.TIME_LENGTH, -1)
            times = torch.tensor(list(range(self.TIME_LENGTH))) / self.TIME_LENGTH
            positional_encoding = times[None, :, None].expand(self.X.shape[0], -1, 1).to(device)
            self.X = torch.cat((self.X, positional_encoding), dim=-1)
        with open(y_filename, "rb") as f:
            self.Y = torch.load(f).to(device)
            self.Y = self.Y.view(self.Y.shape[0], -1, output)

        assert (self.X.shape[0] == self.Y.shape[0])

        self.X_normalise_params = [self.X.mean(dim=[0, 1]), self.X.std(dim=[0, 1])]
        self.Y_normalise_params = [self.Y.mean(dim=[0, 1]), self.Y.std(dim=[0, 1])]
        self.rows = self.X.shape[0]

    def __len__(self):
        return self.rows

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


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