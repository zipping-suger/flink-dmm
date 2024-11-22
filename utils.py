import torch
import torch.nn as nn
import torch.utils.data

import numpy as np
import scipy
import json
import scipy.spatial
import scipy.spatial.distance


def load_data_from_json(filename):
    with open(filename, 'r') as f:
        task_sols = json.load(f)
        
    X = []
    Y = []
   

    for task_sol in task_sols["task_sols"]:
        X.append(task_sol["condensed_task"])
        Y.append(task_sol["solution"])
    
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y

def compute_diff(X_data, Y_data, r, move_r_mult=10**(1/18)):
    x_diff = np.ndarray(X_data.shape[0])
    y_diff = np.ndarray(X_data.shape[0])

    tree = scipy.spatial.KDTree(X_data)
    indices = tree.query_ball_point(X_data, r * move_r_mult, workers=-1)
    for i, index in enumerate(indices):
        x_diff[i] = np.linalg.norm(X_data[index].mean(0) - X_data[i])
        y_diff[i] = np.linalg.norm(Y_data[index].mean(0) - Y_data[i])

    return np.array(x_diff).flatten(), np.array(y_diff).flatten()

def filter_data_dist(X_data, Y_data, r):
    cum = 1
    for i in range(30, 18, -1):
        x_diff, y_diff = compute_diff(X_data, Y_data, r)
        condition = y_diff < i
        X_data, Y_data = X_data[condition], Y_data[condition]
        cum *= condition.mean()
        print(i, cum, condition.mean())
    return X_data, Y_data

def loss_function(outputs, targets):
    # Basic loss (e.g., Mean Squared Error)
    basic_loss = torch.nn.functional.mse_loss(outputs, targets)
    
    return basic_loss

def get_kl_weight(epoch, start, end, max_weight):
    if epoch < start:
        return 0.0
    elif epoch > end:
        return max_weight
    else:
        return max_weight * (epoch - start) / (end - start)

def vae_loss_function(recon_x, x, mu, logvar, kl_weight=1.0):
    # Reconstruction loss (MSE or BCE)
    recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # Combined loss
    return recon_loss + kl_weight * kl_loss

class JointDataset(torch.utils.data.Dataset):
    def __init__(self, x_filename, y_filename, device):
        with open(x_filename, "rb") as f:
            self.X = torch.load(f).to(device)
        with open(y_filename, "rb") as f:
            self.Y = torch.load(f).to(device)
            
        # Convert to float32
        self.X = self.X.float()
        self.Y = self.Y.float()

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


class CVAE(nn.Module):
    def __init__(self, in_size, out_size, in_norm, out_norm, latent_size=32, condition_size=18):
        super(CVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_size + condition_size, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU()
        )
        
        self.latent_size = latent_size
        self.fc_mu = nn.Linear(32, latent_size)
        self.fc_logvar = nn.Linear(32, latent_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size + condition_size, 32),
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

    def encode(self, x, c):
        c_encoded = c
        x = torch.cat([x, c_encoded], dim=1)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-10, max=10) # Clipping logvar to prevent NaNs
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        c_encoded = c
        z = torch.cat([z, c_encoded], dim=1)
        return self.decoder(z)

    def forward(self, x, c):
        x = (x - self.in_norm[0]) / self.in_norm[1]
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        return x_recon * self.out_norm[1] + self.out_norm[0], mu, logvar

    def sample(self, c):
        z = torch.randn(c.shape[0], self.latent_size).to(c.device)
        x_recon = self.decode(z, c)
        return x_recon * self.out_norm[1] + self.out_norm[0]
        

    def to(self, device):
        self.in_norm[0] = self.in_norm[0].to(device)
        self.in_norm[1] = self.in_norm[1].to(device)
        self.out_norm[0] = self.out_norm[0].to(device)
        self.out_norm[1] = self.out_norm[1].to(device)
        return super().to(device)
    
class CVAESampleWrapper(nn.Module):
    def __init__(self, cvae):
        super(CVAESampleWrapper, self).__init__()
        self.cvae = cvae

    def forward(self, c):
        return self.cvae.sample(c)
    
# Convolutional Conditional Variational Autoencoder
class ConvCVAE(nn.Module):
    def __init__(self, in_size, out_size, latent_size=32, condition_size=18, timestep=120):
        super(ConvCVAE, self).__init__()
        self.timestep = timestep

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ELU(),
            nn.Flatten()
        )
        
        self.latent_size = latent_size
        self.fc_mu = nn.Linear(128 * 6 * timestep + condition_size, latent_size)
        self.fc_logvar = nn.Linear(128 * 6 * timestep + condition_size, latent_size)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_size + condition_size, 128 * 6 * timestep)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(3, 3), stride=1, padding=1),
            nn.ELU()
        )
        

        # MLP for encoding the condition
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_size, 32),
            nn.ELU(),
            nn.Linear(32, condition_size)
        )

    def encode(self, x, c):
        c_encoded = self.condition_encoder(c)
        x = x.view(-1, 1, 6, self.timestep)
        x = self.encoder(x)
        x = torch.cat([x, c_encoded], dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-10, max=10) # Clipping logvar to prevent NaNs
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        c_encoded = self.condition_encoder(c)
        z = torch.cat([z, c_encoded], dim=1)
        z = self.decoder_fc(z)
        z = z.view(-1, 128, 6, self.timestep)
        return self.decoder(z).view(-1, 6 * self.timestep)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        return x_recon, mu, logvar

    def sample(self, c):
        z = torch.randn(c.shape[0], self.latent_size).to(c.device)
        x_recon = self.decode(z, c)
        return x_recon 

    

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