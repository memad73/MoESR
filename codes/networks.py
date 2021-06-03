import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SR_Net(nn.Module):
    def __init__(self, channels=3, layers=12, features=64, scale_factor=2):
        super(SR_Net, self).__init__()
        self.scale_factor = scale_factor
        
        model = [nn.Conv2d(channels, features, kernel_size=3, stride=1, padding=1),
                 nn.ReLU(True)]
        
        for i in range(1, layers - 1):
            model += [nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
                      nn.ReLU(True)]
        
        model += [nn.Conv2d(features, channels, kernel_size=3, stride=1, padding=1)]  
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic')
        out = x + self.model(x)  # add skip connections
        return out

  
class ISE_Net(nn.Module):
    def __init__(self, channels=3, layers=6, features=64):
        super(ISE_Net, self).__init__()
        
        model = [nn.Conv2d(channels * 2, features, kernel_size=3, stride=1, padding=0),
                 nn.ReLU(True)]
        
        nf_mult = 1
        nf_mult_prev = 1
        
        for n in range(1, layers - 2):
            model += [nn.Conv2d(features * nf_mult_prev, features * nf_mult, kernel_size=3, stride=2, padding=0),
                      nn.ReLU(True)]
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
        
        model += [nn.Conv2d(features * nf_mult_prev, features * nf_mult, kernel_size=3, stride=1, padding=0)] 
        model += [nn.Conv2d(features * nf_mult, 1, kernel_size=3, stride=1, padding=0)]  

        self.model = nn.Sequential(*model)
        
        
    def forward(self, x, bq):
        net_input = torch.cat([x, bq], 1)
        out = self.model(net_input) 
        return out
      
      
class KEN_Net(nn.Module):
    def __init__(self, n_feature=85, n_hidden=50, n_output=3):
        super(KEN_Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_feature, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )
        
    def forward(self, x):
        # Normalize the input
        mean = x.mean(axis=1)
        std = x.std(axis=1)
        x = (x - mean[:, None]) / std[:, None]
        
        x = self.layers(x)
        x[:, 0:2][x[:, 0:2] < 0.01] = 0.01
        x[:, 2] *= np.pi / 16
        return x

        
def weights_init(m):
    """ initialize weights """
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
            
            
def weights_rotate(m):
    """ rotate network parameters by 90 degree """
    if m.__class__.__name__.find('Conv') != -1:
        m.weight.data = m.weight.data.rot90(1, dims=[2, 3])
        
def weights_flip(m):
    """ flip network parameters """
    if m.__class__.__name__.find('Conv') != -1:
        m.weight.data = m.weight.data.flip(3)
        
def weights_flip_rotate(m):
    """ flip network parameters and then rotate them by 90 degree """
    if m.__class__.__name__.find('Conv') != -1:
        m.weight.data = m.weight.data.flip(3).rot90(dims=[2, 3])