from models import *
from models.process_model import Client_mlp_4
from utils import *
from parse import args


torch.set_default_tensor_type('torch.cuda.FloatTensor')
set_random_seed(1234)

class ImageModel_CVT(nn.Module):
    def __init__(self, dataset, hidden, num_cutlayer, num_classes, device):
        super(ImageModel_CVT, self).__init__()
        
        if args.model == 'resnet':

            if dataset == 'mnist' or dataset == 'fmnist':
                ClientMnist1 = ClientResNet18Mnist(level=args.level, hidden=hidden, num_classes=num_cutlayer).to(device)
                ClientMnist2 = ClientResNet18Mnist(level=args.level, hidden=hidden, num_classes=num_cutlayer).to(device)

                self.client_model_1 = ClientMnist1.GetModel().to(device)
                self.client_model_2 = ClientMnist2.GetModel().to(device)
            else:
                Client1U = ClientResNet18(level=args.level, hidden=hidden, num_classes=num_cutlayer).to(device)
                Client2U = ClientResNet18(level=args.level, hidden=hidden, num_classes=num_cutlayer).to(device)
                Client1C = ClientResNet18(level=args.level, hidden=hidden, num_classes=num_cutlayer).to(device)
                Client2C = ClientResNet18(level=args.level, hidden=hidden, num_classes=num_cutlayer).to(device)

                self.client_model_1U = Client1U.GetModel().to(device)
                self.client_model_1C = Client1C.GetModel().to(device)
                self.client_model_2U = Client2U.GetModel().to(device)
                self.client_model_2C = Client2C.GetModel().to(device)

        self.server_model_partyA = Server_ResNet_cat(hidden2=num_cutlayer * 2, num_classes=num_classes).to(device)
        self.server_model_partyB = Server_ResNet_cat(hidden2=num_cutlayer * 2, num_classes=num_classes).to(device)
        self.server_model_partyAB = Server_ResNet_cat(hidden2=num_cutlayer * 4, num_classes=num_classes).to(device)

    def GetModel(self):
        return self.client_model_1U, self.client_model_1C, self.client_model_2U, self.client_model_2C, \
            self.server_model_partyA, self.server_model_partyB, self.server_model_partyAB

def def_tabular_model_CVT(dataset, level, input_dim1, input_dim2, num_classes, num_cutlayer): 
    
    if level == 1:
        client_model_1U = Client_mlp_2(in_dim=input_dim1, n_hidden_1=64,  n_hidden_2=num_cutlayer).to(device)
        client_model_1C = Client_mlp_2(in_dim=input_dim1, n_hidden_1=64,  n_hidden_2=num_cutlayer).to(device)
        client_model_2U = Client_mlp_2(in_dim=input_dim2, n_hidden_1=64,  n_hidden_2=num_cutlayer).to(device)
        client_model_2C = Client_mlp_2(in_dim=input_dim2, n_hidden_1=64,  n_hidden_2=num_cutlayer).to(device)

    if level == 2:
        client_model_1U = Client_mlp_3(in_dim=input_dim1, n_hidden_1=64,  n_hidden_2=128, n_hidden_3=num_cutlayer).to(device)
        client_model_1C = Client_mlp_3(in_dim=input_dim1, n_hidden_1=64,  n_hidden_2=128, n_hidden_3=num_cutlayer).to(device)
        client_model_2U = Client_mlp_3(in_dim=input_dim2, n_hidden_1=64,  n_hidden_2=128, n_hidden_3=num_cutlayer).to(device)
        client_model_2U = Client_mlp_3(in_dim=input_dim2, n_hidden_1=64,  n_hidden_2=128, n_hidden_3=num_cutlayer).to(device)

    if level == 3:
        client_model_1U = Client_mlp_4(in_dim=input_dim1, n_hidden_1=50,  n_hidden_2=70, n_hidden_3=90, n_hidden_4=num_cutlayer).to(device)
        client_model_1C = Client_mlp_4(in_dim=input_dim1, n_hidden_1=50,  n_hidden_2=70, n_hidden_3=90, n_hidden_4=num_cutlayer).to(device)
        client_model_2U = Client_mlp_4(in_dim=input_dim2, n_hidden_1=50,  n_hidden_2=70, n_hidden_3=90, n_hidden_4=num_cutlayer).to(device)
        client_model_2C = Client_mlp_4(in_dim=input_dim2, n_hidden_1=50,  n_hidden_2=70, n_hidden_3=90, n_hidden_4=num_cutlayer).to(device)


    server_model_partyA = Server_mlp_cat(n_hidden_1=num_cutlayer*2, n_hidden_2=128, n_hidden_3=32, out_dim=num_classes).to(device)
    server_model_partyB = Server_mlp_cat(n_hidden_1=num_cutlayer*2, n_hidden_2=128, n_hidden_3=32, out_dim=num_classes).to(device)
    server_model_partyAB = Server_mlp_cat(n_hidden_1=num_cutlayer*4, n_hidden_2=256, n_hidden_3=64, out_dim=num_classes).to(device)
 
        
    return client_model_1U, client_model_1C, client_model_2U, client_model_2C, \
            server_model_partyA, server_model_partyB, server_model_partyAB

