import os
import copy
from torch import nn
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import *
from utils import *
from parse import args
import matplotlib.pyplot as plt
from Decoder.decoder_image import *
import torch.nn.functional as F

set_random_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
device = torch.device(f"cuda:0")

def get_Missing_item_MIMIC(missing_ratio):
    if missing_ratio == 0:
        return 0, 0, 0, 0
    elif missing_ratio == 0.3:
        return 1, 1, 1, 1
    elif missing_ratio == 0.5:
        return 2, 2, 2, 1
    elif missing_ratio == 0.7:
        return 3, 3, 3, 2
    elif missing_ratio == 1:
        return 4, 4, 4, 3
    
if __name__ == '__main__':
    # Define record path
    root_path = '.'
    data_path = os.path.join(root_path, './Dataset').replace('\\', '/')
    save_path = f'Result/Train_IAVFL_vote_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = open(f'Result/Train_IAVFL_vote_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/A_training_process.txt', 'w+')
    
    # load data
    if args.dataset == 'utkface' or args.dataset == 'celeba':
        train_data_nondataloader, test_data, num_classes, num_classes2, channel, hidden = load_data(args.dataset, args.attributes, data_path, args.batch_size)
    elif args.dataset =='MIMIC':
        train_data_nondataloader, test_data, input_dim1, input_dim2, input_dim3, input_dim4, num_classes = load_data_tabular_4clients(args.dataset, args.batch_size)
    elif args.dataset == "imagenet":
        train_data_nondataloader, test_data, num_classes, channel, hidden = get_tinyimagenet_bothloader(batch_size=args.batch_size, shuffle=True, seed=args.seed, device=device, classes = args.classes)
    else:
        train_data_nondataloader, test_data, num_classes, channel, hidden = gen_dataset(args.dataset, data_path, device = device)

    # define model
    if args.dataset =='cifar10' or args.dataset =='imagenet' or args.dataset =='utkface':
        ImageModel_1= ImageModel_4clients(dataset= args.dataset, hidden=hidden, num_cutlayer=args.num_cutlayer, num_classes=num_classes, mode='standalone', device=device)
        IAVFL_1_Client1_model, IAVFL_1_Client2_model, IAVFL_1_Client3_model, IAVFL_1_Client4_model, IAVFL_1_Server_model = ImageModel_1.GetModel()
        
        ImageModel_2= ImageModel_4clients(dataset= args.dataset, hidden=hidden, num_cutlayer=args.num_cutlayer, num_classes=num_classes, mode='standalone', device=device)
        IAVFL_2_Client1_model, IAVFL_2_Client2_model, IAVFL_2_Client3_model, IAVFL_2_Client4_model, IAVFL_2_Server_model = ImageModel_2.GetModel()
        
        ImageModel_3= ImageModel_4clients(dataset= args.dataset, hidden=hidden, num_cutlayer=args.num_cutlayer, num_classes=num_classes, mode='standalone', device=device)
        IAVFL_3_Client1_model, IAVFL_3_Client2_model, IAVFL_3_Client3_model, IAVFL_3_Client4_model, IAVFL_3_Server_model = ImageModel_3.GetModel()
        
        ImageModel_4= ImageModel_4clients(dataset= args.dataset, hidden=hidden, num_cutlayer=args.num_cutlayer, num_classes=num_classes, mode='standalone', device=device)
        IAVFL_4_Client1_model, IAVFL_4_Client2_model, IAVFL_4_Client3_model, IAVFL_4_Client4_model, IAVFL_4_Server_model = ImageModel_4.GetModel()
        
        ImageModel_5 = ImageModel_4clients(dataset= args.dataset, hidden=hidden, num_cutlayer=int(args.num_cutlayer/4), num_classes=num_classes, mode='cat', device = device)
        VFL_Client1_model, VFL_Client2_model, VFL_Client3_model, VFL_Client4_model, VFL_Server_model = ImageModel_5.GetModel()
    
    elif args.dataset =='MIMIC':
        IAVFL_1_Client1_model, IAVFL_1_Client2_model, IAVFL_1_Client3_model, IAVFL_1_Client4_model, IAVFL_1_Server_model = def_tabular_model_4clients(dataset= args.dataset, level= args.level, input_dim1=input_dim1, input_dim2=input_dim2, input_dim3=input_dim3, input_dim4=input_dim4, num_classes=num_classes, num_cutlayer=args.num_cutlayer, mode='standalone', device=device)
        
        IAVFL_2_Client1_model, IAVFL_2_Client2_model, IAVFL_2_Client3_model, IAVFL_2_Client4_model, IAVFL_2_Server_model = def_tabular_model_4clients(dataset= args.dataset, level= args.level, input_dim1=input_dim1, input_dim2=input_dim2, input_dim3=input_dim3, input_dim4=input_dim4, num_classes=num_classes, num_cutlayer=args.num_cutlayer, mode='standalone', device=device)
        
        IAVFL_3_Client1_model, IAVFL_3_Client2_model, IAVFL_3_Client3_model, IAVFL_3_Client4_model, IAVFL_3_Server_model = def_tabular_model_4clients(dataset= args.dataset, level= args.level, input_dim1=input_dim1, input_dim2=input_dim2, input_dim3=input_dim3, input_dim4=input_dim4, num_classes=num_classes, num_cutlayer=args.num_cutlayer, mode='standalone', device=device)
        
        IAVFL_4_Client1_model, IAVFL_4_Client2_model, IAVFL_4_Client3_model, IAVFL_4_Client4_model, IAVFL_4_Server_model = def_tabular_model_4clients(dataset= args.dataset, level= args.level, input_dim1=input_dim1, input_dim2=input_dim2, input_dim3=input_dim3, input_dim4=input_dim4, num_classes=num_classes, num_cutlayer=args.num_cutlayer, mode='standalone', device=device)
        
        VFL_Client1_model, VFL_Client2_model, VFL_Client3_model, VFL_Client4_model, VFL_Server_model = def_tabular_model_4clients(dataset= args.dataset, level= args.level, input_dim1=input_dim1, input_dim2=input_dim2, input_dim3=input_dim3, input_dim4=input_dim4, num_classes=num_classes, num_cutlayer=int(args.num_cutlayer/4), mode='cat', device=device)
        
    # load model
    # VFL
    VFL_Client1_best = f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{int(args.num_cutlayer/4)}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client1_best.pth'
    VFL_Client2_best = f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{int(args.num_cutlayer/4)}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client2_best.pth'
    VFL_Client3_best = f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{int(args.num_cutlayer/4)}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client3_best.pth'
    VFL_Client4_best = f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{int(args.num_cutlayer/4)}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client4_best.pth'
    VFL_Server_best = f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{int(args.num_cutlayer/4)}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/server_best.pth'
            
    # IAVFL
    IAVFL_1_Client1_best = f'Result/Results_IAVFL_1_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client1_best.pth'
    IAVFL_1_Server_best = f'Result/Results_IAVFL_1_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/server_best.pth'

    IAVFL_2_Client2_best = f'Result/Results_IAVFL_2_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client2_best.pth'
    IAVFL_2_Server_best = f'Result/Results_IAVFL_2_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/server_best.pth'
                
    IAVFL_3_Client3_best = f'Result/Results_IAVFL_3_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client3_best.pth'
    IAVFL_3_Server_best = f'Result/Results_IAVFL_3_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/server_best.pth'

    IAVFL_4_Client4_best = f'Result/Results_IAVFL_4_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client4_best.pth'
    IAVFL_4_Server_best = f'Result/Results_IAVFL_4_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/server_best.pth'
                
    VFL_Client1_model.load_state_dict(torch.load(VFL_Client1_best, map_location=device))
    VFL_Client2_model.load_state_dict(torch.load(VFL_Client2_best, map_location=device))
    VFL_Client3_model.load_state_dict(torch.load(VFL_Client3_best, map_location=device))
    VFL_Client4_model.load_state_dict(torch.load(VFL_Client4_best, map_location=device))
    VFL_Server_model.load_state_dict(torch.load(VFL_Server_best, map_location=device))

    IAVFL_1_Client1_model.load_state_dict(torch.load(IAVFL_1_Client1_best, map_location=device))
    IAVFL_1_Server_model.load_state_dict(torch.load(IAVFL_1_Server_best, map_location=device))
    
    IAVFL_2_Client2_model.load_state_dict(torch.load(IAVFL_2_Client2_best, map_location=device))
    IAVFL_2_Server_model.load_state_dict(torch.load(IAVFL_2_Server_best, map_location=device))
    
    IAVFL_3_Client3_model.load_state_dict(torch.load(IAVFL_3_Client3_best, map_location=device))
    IAVFL_3_Server_model.load_state_dict(torch.load(IAVFL_3_Server_best, map_location=device))
    
    IAVFL_4_Client4_model.load_state_dict(torch.load(IAVFL_4_Client4_best, map_location=device))
    IAVFL_4_Server_model.load_state_dict(torch.load(IAVFL_4_Server_best, map_location=device))
    
    VFL_Client1_model.eval()
    VFL_Client2_model.eval()
    VFL_Client3_model.eval()
    VFL_Client4_model.eval()
    VFL_Server_model.eval()
    
    IAVFL_1_Client1_model.eval()
    IAVFL_1_Server_model.eval()

    IAVFL_2_Client2_model.eval()
    IAVFL_2_Server_model.eval()

    IAVFL_3_Client3_model.eval()
    IAVFL_3_Server_model.eval()

    IAVFL_4_Client4_model.eval()
    IAVFL_4_Server_model.eval()
    
    # ------------------------------------
    if args.dataset =='cifar10':
        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    elif args.dataset =='MIMIC':
        missing_ratios = [0, 0.3, 0.5, 0.7, 1]
        
    correct = [[0] * len(missing_ratios) for _ in range(4)] 

    size = len(test_data.dataset)
    
    for batch_id, batch in enumerate(test_data):
        X, target = batch
        X_1, X_2, X_3, X_4 = split_data_4clients(args.dataset, X)

        if args.dataset == 'utkface' or args.dataset == 'celeba':
            Y_1 = target[0].to(device)
        else:
            Y_1 = target.to(device)

        VFL_fx1_full_embedding = VFL_Client1_model(X_1)
        VFL_fx2_full_embedding = VFL_Client2_model(X_2)
        VFL_fx3_full_embedding = VFL_Client3_model(X_3)
        VFL_fx4_full_embedding = VFL_Client4_model(X_4)
        
        iavfl_fx1_full_embedding = IAVFL_1_Client1_model(X_1)
        _, iavfl_fx1_full_server = IAVFL_1_Server_model(iavfl_fx1_full_embedding)
        
        iavfl_fx2_full_embedding = IAVFL_2_Client2_model(X_2)
        _, iavfl_fx2_full_server = IAVFL_2_Server_model(iavfl_fx2_full_embedding)
        
        iavfl_fx3_full_embedding = IAVFL_3_Client3_model(X_3)
        _, iavfl_fx3_full_server = IAVFL_3_Server_model(iavfl_fx3_full_embedding)
        
        iavfl_fx4_full_embedding = IAVFL_4_Client4_model(X_4)
        _, iavfl_fx4_full_server = IAVFL_4_Server_model(iavfl_fx4_full_embedding)
        
        half_width = int(X.shape[-1]/2)
        half_height = int(X.shape[-2]/2)
        
        if args.dataset == 'cifar10':
            missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        elif args.dataset == 'MIMIC':
            missing_ratios = [0, 0.3, 0.5, 0.7, 1]
            
        for i, missing_ratio in enumerate(missing_ratios):
            
            if missing_ratio > 0:   
                if args.dataset == 'cifar10':  
                    # Client 1
                    X1_blank = X_1.clone()
                    X1_blank[:, :, int(half_height * (1 - missing_ratio)):, int(half_width * (1 - missing_ratio)):] = 0

                    # Client 2
                    X2_blank = X_2.clone()
                    X2_blank[:, :, int(half_height * (1 - missing_ratio)):, :int(half_width * (1 + missing_ratio))] = 0

                    # Client 3
                    X3_blank = X_3.clone()
                    X3_blank[:, :, :int(half_height * (1 + missing_ratio)), int(half_width * (1 - missing_ratio)):] = 0

                    # Client 4
                    X4_blank = X_4.clone()
                    X4_blank[:, :, :int(half_height * (1 + missing_ratio)), :int(half_width * (1 + missing_ratio))] = 0
                elif args.dataset == 'MIMIC':
                    Missing_item_1, Missing_item_2, Missing_item_3, Missing_item_4 = get_Missing_item_MIMIC(missing_ratio)

                    X1_blank = X_1.clone()
                    X1_blank[:,-Missing_item_1:] = 0
                    X2_blank = X_2.clone()
                    X2_blank[:,-Missing_item_2:] = 0
                    X3_blank = X_3.clone()
                    X3_blank[:,-Missing_item_3:] = 0
                    X4_blank = X_4.clone()
                    X4_blank[:,-Missing_item_4:] = 0
                    
            else:
                X1_blank, X2_blank, X3_blank, X4_blank = X_1, X_2, X_3, X_4
            
            # ---client 1---
            fx1_iavfl_1 = IAVFL_1_Client1_model(X1_blank)
            _, iavfl_1_server = IAVFL_1_Server_model(fx1_iavfl_1)
            iavfl_1_server_pre_class = iavfl_1_server.argmax(1)
            
            iavfl_2_server_pre_class = iavfl_fx2_full_server.argmax(1)
            iavfl_3_server_pre_class = iavfl_fx3_full_server.argmax(1)
            iavfl_4_server_pre_class = iavfl_fx4_full_server.argmax(1)
            
            VFL_fx1 = VFL_Client1_model(X1_blank)
            _, VFL_fx1_server = VFL_Server_model(VFL_fx1, VFL_fx2_full_embedding, VFL_fx3_full_embedding, VFL_fx4_full_embedding)
            VFL_fx1_server_pre_class = VFL_fx1_server.argmax(1)
            
            pre_client1 = torch.stack([VFL_fx1_server_pre_class, iavfl_1_server_pre_class, iavfl_2_server_pre_class, iavfl_3_server_pre_class, iavfl_4_server_pre_class], dim=0)
            final_pre_client1, _ = torch.mode(pre_client1, dim=0)
            
            correct[0][i] += (final_pre_client1 == Y_1).type(torch.float).sum().item()
            
            # ---client 2---
            fx2_iavfl_2 = IAVFL_2_Client2_model(X2_blank)
            _, iavfl_2_server = IAVFL_2_Server_model(fx2_iavfl_2)
            iavfl_2_server_pre_class = iavfl_2_server.argmax(1)
            
            iavfl_1_server_pre_class = iavfl_fx1_full_server.argmax(1)
            iavfl_3_server_pre_class = iavfl_fx3_full_server.argmax(1)
            iavfl_4_server_pre_class = iavfl_fx4_full_server.argmax(1)
            
            VFL_fx2 = VFL_Client2_model(X2_blank)
            _, VFL_fx2_server = VFL_Server_model(VFL_fx1_full_embedding, VFL_fx2, VFL_fx3_full_embedding, VFL_fx4_full_embedding)
            VFL_fx2_server_pre_class = VFL_fx2_server.argmax(1)
            
            pre_client2 = torch.stack([VFL_fx2_server_pre_class, iavfl_1_server_pre_class, iavfl_2_server_pre_class, iavfl_3_server_pre_class, iavfl_4_server_pre_class], dim=0)
            final_pre_client2, _ = torch.mode(pre_client2, dim=0)
            
            correct[1][i] += (final_pre_client2 == Y_1).type(torch.float).sum().item()
            
            # ---client 3---
            fx3_iavfl_3 = IAVFL_3_Client3_model(X3_blank)
            _, iavfl_3_server = IAVFL_3_Server_model(fx3_iavfl_3)
            iavfl_3_server_pre_class = iavfl_3_server.argmax(1)
            
            iavfl_1_server_pre_class = iavfl_fx1_full_server.argmax(1)
            iavfl_2_server_pre_class = iavfl_fx2_full_server.argmax(1)
            iavfl_4_server_pre_class = iavfl_fx4_full_server.argmax(1)
            
            VFL_fx3 = VFL_Client3_model(X3_blank)
            _, VFL_fx3_server = VFL_Server_model(VFL_fx1_full_embedding, VFL_fx2_full_embedding, VFL_fx3, VFL_fx4_full_embedding)
            VFL_fx3_server_pre_class = VFL_fx3_server.argmax(1)
            
            pre_client3 = torch.stack([VFL_fx3_server_pre_class, iavfl_1_server_pre_class, iavfl_2_server_pre_class, iavfl_3_server_pre_class, iavfl_4_server_pre_class], dim=0)
            final_pre_client3, _ = torch.mode(pre_client3, dim=0)
            
            correct[2][i] += (final_pre_client3 == Y_1).type(torch.float).sum().item()
            
            # ---client 4---
            fx4_iavfl_4 = IAVFL_4_Client4_model(X4_blank)
            _, iavfl_4_server = IAVFL_4_Server_model(fx4_iavfl_4)
            iavfl_4_server_pre_class = iavfl_4_server.argmax(1)
            
            iavfl_1_server_pre_class = iavfl_fx1_full_server.argmax(1)
            iavfl_2_server_pre_class = iavfl_fx2_full_server.argmax(1)
            iavfl_3_server_pre_class = iavfl_fx3_full_server.argmax(1)
            
            VFL_fx4 = VFL_Client4_model(X4_blank)
            _, VFL_fx4_server = VFL_Server_model(VFL_fx1_full_embedding, VFL_fx2_full_embedding, VFL_fx3_full_embedding, VFL_fx4)
            VFL_fx4_server_pre_class = VFL_fx4_server.argmax(1)
            
            pre_client4 = torch.stack([VFL_fx4_server_pre_class, iavfl_1_server_pre_class, iavfl_2_server_pre_class, iavfl_3_server_pre_class, iavfl_4_server_pre_class], dim=0)
            final_pre_client4, _ = torch.mode(pre_client4, dim=0)
            
            correct[3][i] += (final_pre_client4 == Y_1).type(torch.float).sum().item()
            
        if batch_id == len(test_data) - 1:
            
            if args.dataset == 'cifar10':
                missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
            elif args.dataset == 'MIMIC':
                missing_ratios = [0, 0.3, 0.5, 0.7, 1]
                
            for i, missing_ratio in enumerate(missing_ratios):
                
                print(f"Missing ratio: {missing_ratio}\n", file=filename)
                avg_acc = (correct[0][i]+correct[1][i]+correct[2][i]+correct[3][i])/(4*size)
                client1_acc = correct[0][i]/size
                client2_acc = correct[1][i]/size
                client3_acc = correct[2][i]/size
                client4_acc = correct[3][i]/size

                print(f"Avg Acc: {(100 * avg_acc):>0.3f}%\n", file=filename)
                
                print("Client 1 is Partially missing: ", file=filename)
                print(f"Acc: {(100 * client1_acc):>0.3f}%\n", file=filename)
                
                print("Client 2 is Partially missing: ", file=filename)
                print(f"Acc: {(100 * client2_acc):>0.3f}%\n", file=filename)
                
                print("Client 3 is Partially missing: ", file=filename)
                print(f"Acc: {(100 * client3_acc):>0.3f}%\n", file=filename)
                
                print("Client 4 is Partially missing: ", file=filename)
                print(f"Acc: {(100 * client4_acc):>0.3f}%\n", file=filename)
                
                
                
