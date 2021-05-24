import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from neuralfp.modules.encoder import *

# d = 128
# h = 1024
# u = 32
# v = h/d

class Neuralfp(nn.Module):
    def __init__(self, encoder, n_features):
        super(Neuralfp, self).__init__()
        self.encoder = encoder
        self.n_features = n_features
        self.projector = nn.ModuleList()
        for _ in range(d):
            de_block = nn.Sequential(nn.Linear(v,u),
                                     nn.ELU(),
                                     nn.Linear(u,1)
                                     )
            self.projector.append(de_block)
        

    def forward(self, x_i, x_j):
        
       	h_i = torch.flatten(self.encoder(x_i))
       	splitheads = torch.split(h_i, v)
        for ix, layer in enumerate(self.projector):
            splitheads[ix] = layer(splitheads[ix])
        z_i = F.normalize(torch.cat(splitheads, dim=0), p=2)
        
       	h_j = torch.flatten(self.encoder(x_j))
       	splitheads = torch.split(h_j, v)
        for ix, layer in enumerate(self.projector):
            splitheads[ix] = layer(splitheads[ix])
        z_j = F.normalize(torch.cat(splitheads, dim=0), p=2)
        
        return h_i, h_j, z_i, z_j
        

