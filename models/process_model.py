from models import *
from models.process_model import Client_mlp_4
from utils import *
from parse import args

class ImageModel(nn.Module):
    def __init__(self, dataset, hidden, num_cutlayer, num_classes, mode, device):
        super(ImageModel, self).__init__()

        if args.model == 'resnet':
            if dataset == 'mnist' or dataset == 'fmnist':
                ClientMnist1 = ClientResNet18Mnist(level=args.level, hidden=hidden, num_classes=num_cutlayer).to(device)
                ClientMnist2 = ClientResNet18Mnist(level=args.level, hidden=hidden, num_classes=num_cutlayer).to(device)

                self.client_model_1 = ClientMnist1.GetModel().to(device)
                self.client_model_2 = ClientMnist2.GetModel().to(device)
            else:
                Client1 = ClientResNet18(level=args.level, hidden=hidden, num_classes=num_cutlayer).to(device)
                Client2 = ClientResNet18(level=args.level, hidden=hidden, num_classes=num_cutlayer).to(device)

                self.client_model_1 = Client1.GetModel().to(device)
                self.client_model_2 = Client2.GetModel().to(device)

        if mode == 'cat':
            self.server_model = Server_ResNet_cat(hidden2=num_cutlayer * 2, num_classes=num_classes).to(device)
        elif mode == 'sum':
            self.server_model = Server_ResNet_sum(hidden2=num_cutlayer, num_classes=num_classes).to(device)
        elif mode == 'standalone':
            self.server_model = Server_ResNet_standalone(hidden2=num_cutlayer, num_classes=num_classes).to(device)
        else:
            raise ValueError(f"Mode '{mode}' is not supported")
    
    def GetModel(self):
        return self.client_model_1, self.client_model_2, self.server_model


class ImageModel_4clients(nn.Module):
    def __init__(self, dataset, hidden, num_cutlayer, num_classes, mode, device):
        super(ImageModel_4clients, self).__init__()

        if args.model == 'resnet':
            Client1 = ClientResNet18(level=args.level, hidden=hidden, num_classes=num_cutlayer).to(device)
            Client2 = ClientResNet18(level=args.level, hidden=hidden, num_classes=num_cutlayer).to(device)
            Client3 = ClientResNet18(level=args.level, hidden=hidden, num_classes=num_cutlayer).to(device)
            Client4 = ClientResNet18(level=args.level, hidden=hidden, num_classes=num_cutlayer).to(device)

            self.client_model_1 = Client1.GetModel().to(device)
            self.client_model_2 = Client2.GetModel().to(device)
            self.client_model_3 = Client3.GetModel().to(device)
            self.client_model_4 = Client4.GetModel().to(device)

        if mode == 'cat':
            self.server_model = Server_ResNet_cat_4clients(hidden=num_cutlayer * 4 , num_classes=num_classes).to(device)
        elif mode == 'sum':
            self.server_model = Server_ResNet_sum_4clients(hidden=num_cutlayer, num_classes=num_classes).to(device)
        elif mode == 'standalone':
            self.server_model = Server_ResNet_standalone_4clients(hidden=num_cutlayer, num_classes=num_classes).to(device)
        else:
            raise ValueError(f"Mode '{mode}' is not supported")
    
    def GetModel(self):
        return self.client_model_1, self.client_model_2, self.client_model_3,\
            self.client_model_4, self.server_model
            
# define tabular model for 2 clients
def def_tabular_model(dataset, level, input_dim1, input_dim2, num_classes, num_cutlayer, mode): 
    if level == 1:
        client_model_1 = Client_mlp_2(in_dim=input_dim1, n_hidden_1=int(num_cutlayer/2),  n_hidden_2=num_cutlayer).to(device)
        client_model_2 = Client_mlp_2(in_dim=input_dim2, n_hidden_1=int(num_cutlayer/2),  n_hidden_2=num_cutlayer).to(device)

    if level == 2:
        client_model_1 = Client_mlp_3(in_dim=input_dim1, n_hidden_1=int(num_cutlayer/4),  n_hidden_2=int(num_cutlayer/2), n_hidden_3=num_cutlayer).to(device)
        client_model_2 = Client_mlp_3(in_dim=input_dim2, n_hidden_1=int(num_cutlayer/4),  n_hidden_2=int(num_cutlayer/2), n_hidden_3=num_cutlayer).to(device)

    if level == 3:
        client_model_1 = Client_mlp_4(in_dim=input_dim1, n_hidden_1=64,  n_hidden_2=128, n_hidden_3=256, n_hidden_4=num_cutlayer).to(device)
        client_model_2 = Client_mlp_4(in_dim=input_dim2, n_hidden_1=64,  n_hidden_2=128, n_hidden_3=256, n_hidden_4=num_cutlayer).to(device)
        
    if level == 4:
        client_model_1 = Client_mlp_5(in_dim=input_dim1, n_hidden_1=int(num_cutlayer/8),  n_hidden_2=int(num_cutlayer/8), n_hidden_3=int(num_cutlayer/4), n_hidden_4=int(num_cutlayer/2), n_hidden_5=num_cutlayer).to(device)
        client_model_2 = Client_mlp_5(in_dim=input_dim2, n_hidden_1=int(num_cutlayer/8),  n_hidden_2=int(num_cutlayer/8), n_hidden_3=int(num_cutlayer/4), n_hidden_4=int(num_cutlayer/2), n_hidden_5=num_cutlayer).to(device)

    # num_cutlayer
    if mode == 'cat':
        server_model = Server_mlp_cat(n_hidden_1=num_cutlayer*2, n_hidden_2=128, n_hidden_3=32, out_dim=num_classes).to(device)
    elif mode == 'standalone':
        server_model = Server_mlp_standalone(n_hidden_1=num_cutlayer, n_hidden_2=128, n_hidden_3=32,  out_dim=num_classes).to(device)
    elif mode == 'sum':
        server_model = Server_mlp_sum(n_hidden_1=num_cutlayer, n_hidden_2=128, n_hidden_3=32, out_dim=num_classes).to(device)
        
    return client_model_1, client_model_2, server_model

# define tabular model for 4 clients
def def_tabular_model_4clients(dataset, level, input_dim1, input_dim2, input_dim3, input_dim4, num_classes, num_cutlayer, mode, device): 
    if level == 1:
        client_model_1 = Client_mlp_2(in_dim=input_dim1, n_hidden_1=int(num_cutlayer/2),  n_hidden_2=num_cutlayer).to(device)
        client_model_2 = Client_mlp_2(in_dim=input_dim2, n_hidden_1=int(num_cutlayer/2),  n_hidden_2=num_cutlayer).to(device)
        client_model_3 = Client_mlp_2(in_dim=input_dim3, n_hidden_1=int(num_cutlayer/2),  n_hidden_2=num_cutlayer).to(device)
        client_model_4 = Client_mlp_2(in_dim=input_dim4, n_hidden_1=int(num_cutlayer/2),  n_hidden_2=num_cutlayer).to(device)

    if level == 2:
        client_model_1 = Client_mlp_3(in_dim=input_dim1, n_hidden_1=int(num_cutlayer/4),  n_hidden_2=int(num_cutlayer/2), n_hidden_3=num_cutlayer).to(device)
        client_model_2 = Client_mlp_3(in_dim=input_dim2, n_hidden_1=int(num_cutlayer/4),  n_hidden_2=int(num_cutlayer/2), n_hidden_3=num_cutlayer).to(device)
        client_model_3 = Client_mlp_3(in_dim=input_dim3, n_hidden_1=int(num_cutlayer/4),  n_hidden_2=int(num_cutlayer/2), n_hidden_3=num_cutlayer).to(device)
        client_model_4 = Client_mlp_3(in_dim=input_dim4, n_hidden_1=int(num_cutlayer/4),  n_hidden_2=int(num_cutlayer/2), n_hidden_3=num_cutlayer).to(device)
    
    if level == 3:
        client_model_1 = Client_mlp_4(in_dim=input_dim1, n_hidden_1=8,  n_hidden_2=32, n_hidden_3=128, n_hidden_4=num_cutlayer).to(device)
        client_model_2 = Client_mlp_4(in_dim=input_dim2, n_hidden_1=8,  n_hidden_2=32, n_hidden_3=128, n_hidden_4=num_cutlayer).to(device)
        client_model_3 = Client_mlp_4(in_dim=input_dim3, n_hidden_1=8,  n_hidden_2=32, n_hidden_3=128, n_hidden_4=num_cutlayer).to(device)
        client_model_4 = Client_mlp_4(in_dim=input_dim4, n_hidden_1=8,  n_hidden_2=32, n_hidden_3=128, n_hidden_4=num_cutlayer).to(device)
    
    if level == 4:
        client_model_1 = Client_mlp_5(in_dim=input_dim1, n_hidden_1=int(num_cutlayer/8),  n_hidden_2=int(num_cutlayer/8), n_hidden_3=int(num_cutlayer/4), n_hidden_4=int(num_cutlayer/2), n_hidden_5=num_cutlayer).to(device)
        client_model_2 = Client_mlp_5(in_dim=input_dim2, n_hidden_1=int(num_cutlayer/8),  n_hidden_2=int(num_cutlayer/8), n_hidden_3=int(num_cutlayer/4), n_hidden_4=int(num_cutlayer/2), n_hidden_5=num_cutlayer).to(device)
        client_model_3 = Client_mlp_5(in_dim=input_dim3, n_hidden_1=int(num_cutlayer/8),  n_hidden_2=int(num_cutlayer/8), n_hidden_3=int(num_cutlayer/4), n_hidden_4=int(num_cutlayer/2), n_hidden_5=num_cutlayer).to(device)
        client_model_4 = Client_mlp_5(in_dim=input_dim4, n_hidden_1=int(num_cutlayer/8),  n_hidden_2=int(num_cutlayer/8), n_hidden_3=int(num_cutlayer/4), n_hidden_4=int(num_cutlayer/2), n_hidden_5=num_cutlayer).to(device)

    # num_cutlayer
    if mode == 'cat':
        server_model = Server_mlp_cat_4clients(n_hidden_1=num_cutlayer*4, n_hidden_2=128, n_hidden_3=32, out_dim=num_classes).to(device)
    
    elif mode == 'standalone':
        server_model = Server_mlp_standalone_4clients(n_hidden_1=num_cutlayer, n_hidden_2=128, n_hidden_3=32,  out_dim=num_classes).to(device)

    elif mode == 'sum':
        server_model = Server_mlp_sum_4clients(n_hidden_1=num_cutlayer, n_hidden_2=128, n_hidden_3=32, out_dim=num_classes).to(device)
        
    return client_model_1, client_model_2, client_model_3, client_model_4, server_model
