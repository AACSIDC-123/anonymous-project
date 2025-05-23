
import torch
import torch.nn as nn
import numpy as np
from parse import args
import os

os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
device = torch.device(f"cuda:0") 

class Generator_tabular_1(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=256,  n_hidden_2=2048):
        super(Generator_tabular_1, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_1),
            nn.LeakyReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, in_dim),
        )

        self.in_dim = in_dim
       
    def forward(self, x, x_the_other, Missing_item, batch_size):

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out) 

        
        if Missing_item == self.in_dim:
            layer2_out = layer2_out
        else:
            layer2_out[:,:(self.in_dim - Missing_item)] = x_the_other[:,:(self.in_dim - Missing_item)]
            
        if layer2_out.shape[0] != batch_size:
            raise ValueError("batch size does not match")
        
        # Check for NaN values in the input data
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError("NaN detected in input data")
        
        return layer2_out


class Generator_tabular_2(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=64,  n_hidden_2=512, n_hidden_3=2048):
        super(Generator_tabular_2, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(n_hidden_3, n_hidden_2),
            nn.LeakyReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_1),
            nn.LeakyReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_1, in_dim),
        )

        self.in_dim = in_dim

    def forward(self, x, x_the_other, Missing_item, batch_size):

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out) 
        layer3_out = self.layer3(layer2_out)
        
        if Missing_item == self.in_dim:
            layer3_out = layer3_out
        else:
            layer3_out[:,:(self.in_dim - Missing_item)] = x_the_other[:,:(self.in_dim - Missing_item)]
        
        if layer3_out.shape[0] != batch_size:
            raise ValueError("batch size does not match")
        
        # Check for NaN values in the input data
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError("NaN detected in input data")
        
        return layer3_out


class Generator_mimic_3_4clients(nn.Module):
    def __init__(self, in_dim=4, n_hidden_1=8, n_hidden_2=32, n_hidden_3=128, n_hidden_4=512):
        super(Generator_mimic_3_4clients, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(n_hidden_4, n_hidden_3),
            nn.LeakyReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_3, n_hidden_2),
            nn.LeakyReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_1),
            nn.LeakyReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(n_hidden_1, in_dim)
        )
        self.in_dim = in_dim

    def forward(self, x, x_true, Missing_item, batch_size):
        
        # x is the representation
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out) 
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        
        if Missing_item == self.in_dim:
            layer4_out = layer4_out
        else:
            layer4_out[:,:(self.in_dim - Missing_item)] = x_true[:,:(self.in_dim - Missing_item)]
            
        if layer4_out.shape[0] != batch_size:
            raise ValueError("batch size does not match")
        
        # Check for NaN values in the input data
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError("NaN detected in input data")
        
        return layer4_out
    
class Generator_tabular_3(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=128, n_hidden_2=256, n_hidden_3=512, n_hidden_4=1024):
        super(Generator_tabular_3, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(n_hidden_4, n_hidden_3),
            nn.LeakyReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_3, n_hidden_2),
            nn.LeakyReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_1),
            nn.LeakyReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(n_hidden_1, in_dim)
        )
        self.in_dim = in_dim

    def forward(self, x, x_the_other, Missing_item, batch_size):
        # x is the representation
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out) 
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        
        if Missing_item == self.in_dim:
            layer4_out = layer4_out
        else:
            layer4_out[:,:(self.in_dim - Missing_item)] = x_the_other[:,:(self.in_dim - Missing_item)]
            
        if layer4_out.shape[0] != batch_size:
            raise ValueError("batch size does not match")
        
        # Check for NaN values in the input data
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError("NaN detected in input data")
        
        return layer4_out
    
class Generator_tabular_4(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=32,  n_hidden_2=128, n_hidden_3=512, n_hidden_4=1024, n_hidden_5=2048):
        super(Generator_tabular_4, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(n_hidden_5, n_hidden_4),
            nn.LeakyReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_4, n_hidden_3),
            nn.LeakyReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_3, n_hidden_2),
            nn.LeakyReLU()
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_1),
            nn.LeakyReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Linear(n_hidden_1, in_dim)
        )
        
        self.in_dim = in_dim

    def forward(self, x, x_the_other, Missing_item, batch_size):

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out) 
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        
        
        if Missing_item == self.in_dim:
            layer5_out = layer5_out
        else:
            layer5_out[:,:(self.in_dim - Missing_item)] = x_the_other[:,:(self.in_dim - Missing_item)]
            
        if layer5_out.shape[0] != batch_size:
            raise ValueError("batch size does not match")
        
        # Check for NaN values in the input data
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError("NaN detected in input data")
        
        return layer5_out



