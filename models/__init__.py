
# image
from models.resnet_mnist import ClientResNet18Mnist
from models.resnet import  Server_ResNet_cat, Server_ResNet_cat_4clients, \
    Server_ResNet_sum, Server_ResNet_sum_4clients, Server_ResNet_standalone, Server_ResNet_standalone_4clients, ClientResNet18

#tabular
from models.MLP_tabular import Client_mlp_2, Client_mlp_3,Client_mlp_4, Client_mlp_5, Server_mlp_cat, Server_mlp_standalone, Server_mlp_sum, \
    Server_mlp_cat_4clients, Server_mlp_standalone_4clients, Server_mlp_sum_4clients  

# define -process
from models.process_model import  def_tabular_model, def_tabular_model_4clients, ImageModel, ImageModel_4clients

# FedCVT
from models.process_model_CVT import  def_tabular_model_CVT, ImageModel_CVT
