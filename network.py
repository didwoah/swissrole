import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t, embedding_dim, max_period=10000):
        half_dim = embedding_dim // 2
        exponent = -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        exponent = exponent.to(t.device)
        sinusoid_arg = t[:, None].float() * torch.exp(exponent)[None, :]
        
        embedding = torch.cat([torch.sin(sinusoid_arg), torch.cos(sinusoid_arg)], dim=-1)
        if embedding_dim % 2 == 1: 
            embedding = torch.nn.functional.pad(embedding, (0, 1, 0, 0))
        return embedding

    def forward(self, t):
        if t.ndim == 0:
            t = t.unsqueeze(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LinearBlock(nn.Module):
    def __init__(self, dim_in, dim_out, num_timesteps):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_timesteps = num_timesteps

        self.time_embedding = TimeEmbedding(dim_out)
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x, t):
        x = self.fc(x)
        emb = self.time_embedding(t).view(-1, self.dim_out)

        return emb * x
    
class CustomNet(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hids, num_timesteps):
        super().__init__()
        layers = []
        layers.append(LinearBlock(dim_in, dim_hids[0], num_timesteps))
        prev_dim_hids = dim_hids[0]
        for dim_hid in dim_hids[1:]:
            layers.append(LinearBlock(prev_dim_hids, dim_hid, num_timesteps))
            prev_dim_hids = dim_hid
        layers.append(LinearBlock(prev_dim_hids, dim_out, num_timesteps))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, t):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, t))
        x = self.layers[-1](x, t)
        return x
