import torch
import torch.nn as nn

d = 128
h = 1024
u = 32
v = h/d
chang_fp = [d,d,2*d,2*d,4*d,4*d,h,h]

class Encoder(nn.Module):
	def __init__(self, in_channels=1, stride=2, kernel_size=3):
		super(Encoder, self).__init__()
		self.in_channels = in_channels
		self.stride = stride
		self.kernel_size = kernel_size
		self.conv_layers = self.create_conv_layers(chang_fp)
		
	def forward(self, x):
		x = self.conv_layers(x)
		
	
	def create_conv_layers(self, architecture):
		layers = []
		in_channels = self.in_channels
		kernel_size = self.kernel_size
		stride = self.stride
		
		for x in architecture:
			out_channels = x
			layers.append(nn.Conv2d(in_channels, out_channels, (1,kernel_size), (1,stride), padding=0))
			layers.append(nn.LayerNorm(out_channels))
			layers.append(nn.ReLU())
			layers.append(nn.Conv2d(in_channels, out_channels, (kernel_size,1), (stride,1), padding=0))
			layers.append(nn.LayerNorm(out_channels))
			layers.append(nn.ReLU())

		return nn.Sequential(*layers)
			
			

