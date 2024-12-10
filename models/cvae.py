import torch
import torch.nn as nn

def vae_loss_function(recon_x, x, mu, logvar, kl_weight=1.0):
    # Reconstruction loss (MSE or BCE)
    recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # Combined loss
    return recon_loss + kl_weight * kl_loss

class CVAESampleWrapper(nn.Module):
    def __init__(self, cvae):
        super(CVAESampleWrapper, self).__init__()
        self.cvae = cvae

    def forward(self, c):
        return self.cvae.sample(c)

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
