#  Auto regressive MLP model

import torch
import torch.nn as nn

class ARMLPWrapper(nn.Module):
    def __init__(self, armlp, horizon_length = 120):
        super(ARMLPWrapper, self).__init__()
        self.armlp = armlp
        self.horizon_length = horizon_length

    def forward(self, inputs):
        """
        Predict horizon_length steps into the future using an autoregressive model
        """
        preds = []
        x = inputs
        preds.append(x[:, -6:])  # append the last 6 values of the input
        for t in range(self.horizon_length-1):
            time_step = torch.full((inputs.size(0), 1), t, dtype=torch.float32).to(inputs.device)
            pred = x[:, -6:] + self.armlp(x, time_step) # predict the change in state
            preds.append(pred)
            x = torch.cat((x[:,:-6], pred), dim=1) # replace the last 6 values with the predicted values
        traj = torch.stack(preds, dim=1)
        traj = traj.view(inputs.size(0), -1)  #flatten the list of predictions
        return traj

class TimeEmbedding(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.linear1 = nn.Linear(1, embed_size)
        self.linear2 = nn.Linear(embed_size, embed_size)

    def forward(self, t):
        t = t.unsqueeze(-1).float()
        t = self.linear1(t)
        t = torch.sin(t)
        t = self.linear2(t)
        return t

class ARMLP(nn.Module):
    def __init__(self, in_size, out_size, time_embed_size):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_embed_size)
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_size + time_embed_size, 32),
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

    def forward(self, x, t):
        t_embed = self.time_embedding(t).squeeze(1)
        x = self.layers(torch.cat((x, t_embed), -1))
        return x

       