
from torch import nn
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

class Client_mlp_2(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=128, n_hidden_2=256):
        super(Client_mlp_2, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1),
                                    nn.LeakyReLU())


        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    )

    def forward(self, x):

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer2_out


class Client_mlp_3(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=128, n_hidden_2=256, n_hidden_3=512):
        super(Client_mlp_3, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1),
                                    nn.LeakyReLU())

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU())
        
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    )   

    def forward(self, x):
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer3_out


class Client_mlp_4(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=128, n_hidden_2=256, n_hidden_3=512, n_hidden_4=1024):
        super(Client_mlp_4, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.LeakyReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.LeakyReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_3),
            nn.LeakyReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(n_hidden_3, n_hidden_4),
            # nn.LeakyReLU()
        )

    def forward(self, x):
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
           # raise ValueError("NaN detected in input data")

        return layer4_out

class Client_mlp_5(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=128, n_hidden_2=256, n_hidden_3=512, n_hidden_4=1024, n_hidden_5=2048):
        super(Client_mlp_5, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.LeakyReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.LeakyReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_3),
            nn.LeakyReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(n_hidden_3, n_hidden_4),
            nn.LeakyReLU()
        )
        
        self.layer5 = nn.Sequential(
            nn.Linear(n_hidden_4, n_hidden_5),
            nn.LeakyReLU()
        )

    def forward(self, x):
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
           # raise ValueError("NaN detected in input data")
            
        return layer5_out



class Server_mlp_cat(nn.Module):
    def __init__(self, n_hidden_1=256, n_hidden_2=128, n_hidden_3=64, out_dim=2):
        super(Server_mlp_cat, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_3, out_dim),
                                    )                   
        
    def forward(self, x1, x2):
        x= torch.cat([x1, x2], dim=1)
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer2_out, layer3_out


class Server_mlp_cat_4clients(nn.Module):
    def __init__(self, n_hidden_1=256, n_hidden_2=128, n_hidden_3=64, out_dim=2):
        super(Server_mlp_cat_4clients, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_3, out_dim),
                                    )                   
        
    def forward(self, x1, x2, x3, x4):
        x= torch.cat([x1, x2, x3, x4], dim=1)
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer2_out, layer3_out


class Server_mlp_standalone(nn.Module):
    def __init__(self, n_hidden_1=256, n_hidden_2=128, n_hidden_3=64, out_dim=2):
        super(Server_mlp_standalone, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_3, out_dim),
                                   )

    def forward(self, x1, x2):
        layer1_out = self.layer1(x1)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        if np.isnan(np.sum(x1.data.cpu().numpy())):
            raise ValueError()
        return layer2_out, layer3_out
  
  
class Server_mlp_standalone_4clients(nn.Module):
    def __init__(self, n_hidden_1=256, n_hidden_2=128, n_hidden_3=64, out_dim=2):
        super(Server_mlp_standalone_4clients, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_3, out_dim),
                                   )

    def forward(self, x1):
        layer1_out = self.layer1(x1)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        if np.isnan(np.sum(x1.data.cpu().numpy())):
            raise ValueError()
        return layer2_out, layer3_out    

class Server_mlp_sum(nn.Module):
    def __init__(self, n_hidden_1=256, n_hidden_2=128, n_hidden_3=64, out_dim=2):
        super(Server_mlp_sum, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_3, out_dim),
                                   )

    def forward(self, x1, x2):
        x = (x1 + x2)/2
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        
        if np.isnan(np.sum(x1.data.cpu().numpy())):
            raise ValueError()
        
        return layer2_out, layer3_out

class Server_mlp_sum_4clients(nn.Module):
    def __init__(self, n_hidden_1=256, n_hidden_2=128, n_hidden_3=64, out_dim=2):
        super(Server_mlp_sum_4clients, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_3, out_dim),
                                   )

    def forward(self, x1, x2, x3, x4):
        x = (x1 + x2 + x3 + x4)/4
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        
        if np.isnan(np.sum(x1.data.cpu().numpy())):
            raise ValueError()
        
        return layer2_out, layer3_out
