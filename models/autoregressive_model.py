import torch
import torch.nn as nn
import numpy as np
from models.attention import Seq_Transformer

class AutoregressiveModel(nn.Module):
    def __init__(self, configAR, device):
        super(AutoregressiveModel, self).__init__()
        self.num_channels = configAR["output_dims"]
        self.timestep = configAR["timesteps"]
        self.Wk = nn.ModuleList([nn.Linear(configAR["context_dims"], self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax(dim=1) # Check whether correct dimension
        self.device = device
        
        self.projection_head = nn.Sequential(
            nn.Linear(configAR["context_dims"], configAR["output_dims"] // 2),
            nn.BatchNorm1d(configAR["output_dims"] // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configAR["output_dims"] // 2, configAR["output_dims"] // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configAR["context_dims"], depth=4, heads=4, mlp_dim=64)

    def forward(self, z1, z2): # z1, z2 is (batch_size, seq_len, #channels)
        seq_len = z1.shape[1]
        batch = z1.shape[0]

        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # randomly pick time stamps

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z2[:, t_samples + i, :].view(batch, self.num_channels)  # (timestep, batch_size, #channels)
        forward_seq = z1[:, :t_samples + 1, :] # (batch_size, seq_len_sample, #channels)

        c_t = self.seq_transformer(forward_seq) # (batch_size, dimensions_c)

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t) # (timestep, batch_size, #channels)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1)) # (batch_size, batch_size)
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        return nce # , self.projection_head(c_t)