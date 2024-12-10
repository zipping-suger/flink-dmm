import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import torch.nn.utils.spectral_norm as spectral_norm


from models.helpers import (
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
)


def compute_discriminator_loss(discriminator, real_samples, fake_samples, condition, device):
    """Compute discriminator loss using log probabilities as per the formula"""
    # Get discriminator outputs
    d_real = discriminator(condition, real_samples)
    d_fake = discriminator(condition, fake_samples)
    
    # Compute log probabilities
    log_prob_real = torch.log(d_real + 1e-15)  # Add small epsilon to prevent log(0)
    log_prob_fake = torch.log(1 - d_fake +1e-15)
    
    # Average over batch
    loss = -(torch.mean(log_prob_real) + torch.mean(log_prob_fake))
    
    return loss, torch.mean(d_real), torch.mean(d_fake)

def compute_generator_loss(discriminator, fake_samples, condition):
    """Compute generator loss using log probabilities as per the formula"""
    d_fake = discriminator(condition, fake_samples)
    loss = -torch.mean(torch.log(d_fake + 1e-15))
    return loss


class CGANSampleWrapper(nn.Module):
    def __init__(self, generator):
        super(CGANSampleWrapper, self).__init__()
        self.generator = generator
    def forward(self, inputs):
        input_emb = inputs
        # noise to cpu
        noise = torch.randn(inputs.size(0), self.generator.latent_size).to("cpu")
        gen_trajs = self.generator(input_emb, noise)
        return gen_trajs

class VanillaGenerator(nn.Module):
    def __init__(self, emb_size, output_size, latent_size):
        super(VanillaGenerator, self).__init__()
        self.latent_size = latent_size

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat), nn.LeakyReLU(0.2, inplace=True)]
            return layers

        self.model = nn.Sequential(
            *block(latent_size + emb_size, 256),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, 1024),
            *block(1024, 1024),
            nn.Linear(1024, output_size),
        )

    def forward(self, input_emb, noise):
        gen_input = torch.cat((input_emb, noise), -1)
        return self.model(gen_input)


class VanillaDiscriminator(nn.Module):
    def __init__(self, emb_size, output_size):
        super(VanillaDiscriminator, self).__init__()

        self.emb_size = emb_size
        self.emb_expansion_size = 256
        
        self.emb_expansion = nn.Sequential(
        nn.Linear(emb_size, self.emb_expansion_size),  # Expand emb_size
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(self.emb_expansion_size, self.emb_expansion_size),  # Expand emb_size
)

        self.backbone = nn.Sequential(
            nn.Linear(output_size + self.emb_expansion_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            # nn.Linear(512, 256),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input_emb, traj):
        emb_expanded = self.emb_expansion(input_emb)
        x = torch.cat((emb_expanded, traj), -1)
        return self.backbone(x)
    
    
class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=False,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=dim, horizon=horizon),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=dim, horizon=horizon)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=dim, horizon=horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=dim, horizon=horizon),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, cond):
        '''
            x : [ batch x horizon x transition ]
            cond : [ batch x cond_dim ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.cond_mlp(cond)
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x


class UnetGenerator(nn.Module):
    def __init__(self, emb_size: int, time_horizon: int = 120, channels: int = 6):
        super(UnetGenerator, self).__init__()
        self.latent_size = channels * time_horizon
        self.time_horizon = time_horizon
        self.channels = channels

        self.unet = TemporalUnet(
            horizon=time_horizon,
            transition_dim=channels,
            cond_dim=emb_size,
            dim=128,
            dim_mults=(1, 2, 4, 8),
            attention=True
        )

    def forward(self, conditions: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # reshape noise to (batch_size, channels, time_horizon)
        noise = noise.reshape(noise.size(0), self.time_horizon, self.channels)
        x = self.unet(noise, conditions)
        # Flatten the output to (batch_size, channels * time_horizon)
        return x.reshape(x.size(0), -1)
    
    
class ConvDiscriminator(nn.Module):

    def __init__(
        self,
        cond_dim,
        horizon = 120,
        transition_dim = 6,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()
        self.time_horizon = horizon
        self.channels = transition_dim

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.condition_mlp = nn.Sequential(
            nn.Linear(cond_dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=dim, horizon=horizon),
                Downsample1d(dim_out)
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4
        ##
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim_2, kernel_size=5, embed_dim=dim, horizon=horizon)
        self.mid_down1 = Downsample1d(mid_dim_2)
        horizon = horizon // 2
        ##
        self.mid_block2 = ResidualTemporalBlock(mid_dim_2, mid_dim_3, kernel_size=5, embed_dim=dim, horizon=horizon)
        self.mid_down2 = Downsample1d(mid_dim_3)
        horizon = horizon // 2
        ##
        fc_dim = mid_dim_3 * max(horizon, 1)

        # self.final_block = nn.Sequential(
        #     nn.Linear(fc_dim + dim, fc_dim // 2),
        #     nn.Mish(),
        #     nn.Linear(fc_dim // 2, out_dim),
        #     nn.Sigmoid()
        # )
        
        self.final_block = nn.Sequential(
            nn.Linear(160, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
            nn.Sigmoid()
        )

    def forward(self, cond, x, *args):
        '''
            x : [ batch x horizon x transition ]
        '''
        # reshape traj to (batch_size, time_horizon, channels)
        x = x.view(x.size(0), self.time_horizon, self.channels)
        x = einops.rearrange(x, 'b h t -> b t h')

        ## mask out first conditioning timestep, since this is not sampled by the model
        # x[:, :, 0] = 0

        t = self.condition_mlp(cond)

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        ##
        x = self.mid_block1(x, t)
        x = self.mid_down1(x)
        ##
        x = self.mid_block2(x, t)
        x = self.mid_down2(x)
        ##
        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out
    
    

