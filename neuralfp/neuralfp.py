import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from neuralfp.modules.encoder import *

# d = 128
# h = 1024
# u = 32
# v = int(h/d)

class Neuralfp(nn.Module):
    def __init__(self, encoder):
        super(Neuralfp, self).__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(nn.Linear(v,u),
                                       nn.ELU(),
                                       nn.Linear(u,1)
                                       )


    def forward(self, x_i, x_j):
        
       	h_i = self.encoder(x_i)
        new_h_i = h_i.view(h_i.size(0), -1, v)
        z_i = self.projector(new_h_i).squeeze(-1)
        z_i = F.normalize(z_i, p=2)
        
       	h_j = self.encoder(x_j)
        new_h_j = h_j.view(h_j.size(0), -1, v)
        z_j = self.projector(new_h_j).squeeze(-1)
        z_j = F.normalize(z_j, p=2)
        
        return h_i, h_j, z_i, z_j


