import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
import torch.nn.functional


d = 128
h = 1024
u = 32
v = int(h/d)
chang_fp = [d,d,2*d,2*d,4*d,4*d,h,h]

class Encoder(nn.Module):
    def __init__(self, in_channels=1, stride=2, kernel_size=3, si_cnn=False):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.si_cnn = si_cnn
        self.conv_layers = self.create_conv_layers(chang_fp, si_cnn)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0],-1)
        return x
        
    
    def create_conv_layers(self, architecture, si_cnn):
        layers = []
        in_channels = self.in_channels
        kernel_size = self.kernel_size
        stride = self.stride
        shape = [1,256,32]
        for i,channels in enumerate(architecture):
            
            if si_cnn and i==0:
                layers.append(SI_Conv(in_channels=in_channels, out_channels=channels, kernel_size=[kernel_size,kernel_size], stride=[stride,stride], padding=[1,1]))
                shape[0] = channels
                shape[2] = int(np.ceil(shape[2]/2))
                # layers.append(nn.LayerNorm(shape))
                # layers.append(nn.ReLU())
                # layers.append(SI_Conv(in_channels=in_channels, out_channels=channels, kernel_size=[kernel_size,1], stride=[stride,1], padding=[1,0]))
                shape[1] = int(np.ceil(shape[1]/2))
                layers.append(nn.LayerNorm(shape))
                layers.append(nn.ReLU())
                
                in_channels = channels

            else:
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=(1,kernel_size), stride=(1,stride), padding=(0,1)))
                shape[0] = channels
                shape[2] = int(np.ceil(shape[2]/2))
                layers.append(nn.LayerNorm(shape))
                layers.append(nn.ReLU())
                layers.append(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(kernel_size,1), stride=(stride,1), padding=(1,0)))
                shape[1] = int(np.ceil(shape[1]/2))
                layers.append(nn.LayerNorm(shape))
                layers.append(nn.ReLU())
                
                in_channels = channels


        return nn.Sequential(*layers)
    
class SI_Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 padding, dilation=1, mode=1, scale_range=np.arange(1.0,3.1,0.4)):
        super(SI_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale_range = scale_range
        self.dilation = dilation
        self.mode = mode
        self.bias = None
        self.weight = Parameter(torch.Tensor(
                self.out_channels, self.in_channels, *self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)


    def _apply(self, func):
        super(SI_Conv, self)._apply(func)


    def forward(self, input):
        outputs = []
        for i in range(len(self.scale_range)):
            
            input_ups = F.interpolate(input, scale_factor=self.scale_range[i], mode='bilinear', align_corners=True)
            input_conv = F.conv2d(input_ups, self.weight, None, self.stride, self.padding, self.dilation)
            if i==0:
                req_size = list(input_conv.data.shape[2:4])
            out = F.interpolate(input_conv, size = req_size, mode='bilinear', align_corners=True)           
            outputs.append(out.unsqueeze(-1))

        strength, _ = torch.max(torch.cat(outputs, -1), -1)
        return F.relu(strength)

