import os
import copy
from torch import nn
import numpy as np
from models import *
from utils import *
from parse import args
import matplotlib.pyplot as plt
from Decoder.decoder_image import *
import torch.nn.functional as F

set_random_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
device = torch.device(f"cuda:0")

if __name__ == '__main__':
    # Define record path
    root_path = '.'
    data_path = os.path.join(root_path, 'Dataset').replace('\\', '/')
    save_path = f'Result/Results_vote_for_IAVFL/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = open(f'Result/Results_vote_for_IAVFL/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/A_training_process.txt', 'w+')
    
    # load data
    if args.dataset == 'utkface' or args.dataset == 'celeba':
        train_data_nondataloader, test_data, num_classes, num_classes2, channel, hidden = load_data(args.dataset, args.attributes, data_path, args.batch_size)
    elif args.dataset =='MIMIC' or args.dataset == 'avazu' or args.dataset == 'bank':
        train_data_nondataloader, test_data, input_dim1, input_dim2, num_classes = load_data_tabular(args.dataset, args.batch_size)
    elif args.dataset == "imagenet":
        train_data_nondataloader, test_data, num_classes, channel, hidden = get_tinyimagenet_bothloader(batch_size=args.batch_size, shuffle=True, seed=args.seed, device=device, classes = args.classes)    
    else:
        train_data_nondataloader, test_data, num_classes, channel, hidden = gen_dataset(args.dataset, data_path, device = device)

    # define model
    if args.dataset =='MIMIC' or args.dataset == 'avazu' or args.dataset == 'bank':
        IAVFL_1_Client1_model, IAVFL_1_Client2_model, IAVFL_1_Server_model = def_tabular_model(dataset= args.dataset, level= args.level, input_dim1=input_dim1, input_dim2=input_dim2, num_classes=num_classes, num_cutlayer=args.num_cutlayer, mode='standalone')
        IAVFL_2_Client1_model, IAVFL_2_Client2_model, IAVFL_2_Server_model = def_tabular_model(dataset= args.dataset, level= args.level, input_dim1=input_dim1, input_dim2=input_dim2, num_classes=num_classes, num_cutlayer=args.num_cutlayer, mode='standalone')
        VFL_Client1_model, VFL_Client2_model, VFL_Server_model = def_tabular_model(dataset= args.dataset, level= args.level, input_dim1=input_dim1, input_dim2=input_dim2, num_classes=num_classes, num_cutlayer=int(args.num_cutlayer/2), mode='cat')
    else: 
        ImageModel1= ImageModel(dataset= args.dataset, hidden=hidden, num_cutlayer=args.num_cutlayer, num_classes=num_classes, mode='standalone', device=device)
        IAVFL_1_Client1_model, IAVFL_1_Client2_model, IAVFL_1_Server_model = ImageModel1.GetModel()
        
        ImageModel2= ImageModel(dataset= args.dataset, hidden=hidden, num_cutlayer=args.num_cutlayer, num_classes=num_classes, mode='standalone', device=device)
        IAVFL_2_Client1_model, IAVFL_2_Client2_model, IAVFL_2_Server_model = ImageModel2.GetModel()

        ImageModel3= ImageModel(dataset= args.dataset, hidden=hidden, num_cutlayer=int(args.num_cutlayer/2), num_classes=num_classes, mode='cat', device=device)
        VFL_Client1_model, VFL_Client2_model, VFL_Server_model = ImageModel3.GetModel()

    # load model
    # VFL
    VFL_Client1_best = f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{int(args.num_cutlayer/2)}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client1_best.pth'
    VFL_Client2_best = f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{int(args.num_cutlayer/2)}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client2_best.pth'
    VFL_Server_best  = f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{int(args.num_cutlayer/2)}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/server_best.pth'
    
    # IAVFL
    IAVFL_1_Client1_best = f'Result/Results_IAVFL_1/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client1_best.pth'
    IAVFL_1_Client2_best = f'Result/Results_IAVFL_1/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client2_best.pth'
    IAVFL_1_Server_best  = f'Result/Results_IAVFL_1/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/server_best.pth'
            
    IAVFL_2_Client1_best = f'Result/Results_IAVFL_2/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client1_best.pth'
    IAVFL_2_Client2_best = f'Result/Results_IAVFL_2/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client2_best.pth'
    IAVFL_2_Server_best  = f'Result/Results_IAVFL_2/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/server_best.pth'
    
    IAVFL_1_Client1_model.load_state_dict(torch.load(IAVFL_1_Client1_best, map_location=device))
    IAVFL_1_Client2_model.load_state_dict(torch.load(IAVFL_1_Client2_best, map_location=device))
    IAVFL_1_Server_model.load_state_dict(torch.load(IAVFL_1_Server_best, map_location=device))
    
    IAVFL_2_Client1_model.load_state_dict(torch.load(IAVFL_2_Client1_best, map_location=device))
    IAVFL_2_Client2_model.load_state_dict(torch.load(IAVFL_2_Client2_best, map_location=device))
    IAVFL_2_Server_model.load_state_dict(torch.load(IAVFL_2_Server_best, map_location=device))
    
    VFL_Client1_model.load_state_dict(torch.load(VFL_Client1_best, map_location=device))
    VFL_Client2_model.load_state_dict(torch.load(VFL_Client2_best, map_location=device))
    VFL_Server_model.load_state_dict(torch.load(VFL_Server_best, map_location=device))
    
    VFL_Client1_model.eval()
    VFL_Client2_model.eval()
    VFL_Server_model.eval()
    
    IAVFL_1_Client1_model.eval()
    # IAVFL_1_Client2_model.eval()
    IAVFL_1_Server_model.eval()

    # IAVFL_2_Client1_model.eval()
    IAVFL_2_Client2_model.eval()
    IAVFL_2_Server_model.eval()
    
    correct_normal = 0
    correct_basedA = [0] * 7
    correct_basedB = [0] * 7 
    
    size = len(test_data.dataset)
    
    for batch_id, batch in enumerate(test_data):
        X, target = batch
        X_1, X_2 = split_data(args.dataset, X)

        if args.dataset == 'utkface' or args.dataset == 'celeba':
            Y_1 = target[0].to(device)

        else:
            Y_1 = target.to(device)
            
        # normal case:
        # VFL:
        VFL_fx1 = VFL_Client1_model(X_1)
        VFL_fx2 = VFL_Client2_model(X_2)
        _, VFL_server_normal = VFL_Server_model(VFL_fx1, VFL_fx2)
        VFL_predicted_class = VFL_server_normal.argmax(1)
        
        # IAVFL_1
        IAVFL_1_fx1 = IAVFL_1_Client1_model(X_1)
        _, IAVFL_1_normal = IAVFL_1_Server_model(IAVFL_1_fx1, IAVFL_1_fx1)
        IAVFL_1_predicted_class = IAVFL_1_normal.argmax(1)
        
        # IAVFL_2
        IAVFL_2_fx2 = IAVFL_2_Client2_model(X_2)
        _, IAVFL_2_normal = IAVFL_2_Server_model(IAVFL_2_fx2, IAVFL_2_fx2)
        IAVFL_2_predicted_class = IAVFL_2_normal.argmax(1)
        
        predictions_normal = torch.stack([VFL_predicted_class, IAVFL_1_predicted_class, IAVFL_2_predicted_class], dim=0)  # Shape: (3, batch_size)
        final_predictions_normal, _ = torch.mode(predictions_normal, dim=0) 
        
        correct_normal += (final_predictions_normal == Y_1).type(torch.float).sum().item()
        

        # Missings case 1: A is completed and B is completed by zero
        
        VFL_fx1 = VFL_Client1_model(X_1)
        
        IAVFL_1_fx1 = IAVFL_1_Client1_model(X_1)
        
        VFL_fx2 = VFL_Client2_model(X_2)
        
        IAVFL_2_fx2 = IAVFL_2_Client2_model(X_2)
        
        _, IAVFL_1_normal = IAVFL_1_Server_model(IAVFL_1_fx1, IAVFL_1_fx1)
        IAVFL_1_predicted_class = IAVFL_1_normal.argmax(1)
        
        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        
        for i, missing_ratio in enumerate(missing_ratios):
            
            if args.dataset == 'cifar10' or args.dataset == "imagenet" or args.dataset == 'utkface':
            
                num_zeros = int(X.shape[-1]/2)
                shape = list(range(num_zeros, int(num_zeros + num_zeros * missing_ratio))) 
                index = torch.tensor(shape).to(device)

                if index.numel() > 0:
                    X_2_blank = X_2.index_fill(3, index, 0)
                else:
                    X_2_blank = X_2
                    
            
            else:
                if missing_ratio == 0:
                    X_2_blank = X_2
                else:
                    if args.dataset == 'MIMIC':
                        missing_ratio_to_item_B = {
                            0.1: 2,
                            0.3: 3,
                            0.5: 4,
                            0.7: 5,
                            0.9: 6,
                            1.0: 7
                        }
                        Missing_item_B = missing_ratio_to_item_B.get(missing_ratio, None)  

                        X_2[:,-Missing_item_B:] = 0
                        X_2_blank = X_2
                        
                    elif args.dataset == 'bank':

                        missing_ratio_to_item_B = {
                            0.1: 1,
                            0.3: 2,
                            0.5: 3,
                            0.7: 4,
                            0.9: 5,
                            1.0: 28
                        }
                        Missing_item_B = missing_ratio_to_item_B.get(missing_ratio, None) 

                        X_2[:,-Missing_item_B:] = 0
                        X_2_blank = X_2
                        
                    elif args.dataset == 'avazu':
                
                        missing_ratio_to_item_B = {
                            0.1: 1,
                            0.3: 2,
                            0.5: 3,
                            0.7: 13,
                            0.9: 23,
                            1.0: 38
                        }
                        Missing_item_B = missing_ratio_to_item_B.get(missing_ratio, None) 
                        
                        X_2[:,-Missing_item_B:] = 0
                        X_2_blank = X_2
                
            VFL_fx2_blank = VFL_Client2_model(X_2_blank)
            _, VFL_server_basedA = VFL_Server_model(VFL_fx1, VFL_fx2_blank)
            VFL_predicted_class_basedA = VFL_server_basedA.argmax(1)
            
            IAVFL_2_fx2_blank = IAVFL_2_Client2_model(X_2_blank)
            _, IAVFL_2_basedA = IAVFL_2_Server_model(IAVFL_2_fx2_blank, IAVFL_2_fx2_blank)
            IAVFL_2_predicted_class_basedA = IAVFL_2_basedA.argmax(1)
            
            predictions_basedA = torch.stack([VFL_predicted_class_basedA, IAVFL_1_predicted_class, IAVFL_2_predicted_class_basedA], dim=0)  # Shape: (3, batch_size)
            final_predictions_basedA, _ = torch.mode(predictions_basedA, dim=0) 

            correct_basedA[i] += (final_predictions_basedA == Y_1).type(torch.float).sum().item()
        
        
        # Missings case 2: B is completed and A is completed by zero

        _, IAVFL_2_normal = IAVFL_2_Server_model(IAVFL_2_fx2, IAVFL_2_fx2)
        IAVFL_2_predicted_class = IAVFL_2_normal.argmax(1)
        
        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        
        for i, missing_ratio in enumerate(missing_ratios):
            
            if args.dataset == 'cifar10' or args.dataset == "imagenet" or args.dataset == 'utkface':
            
                num_zeros = int(X.shape[-1]/2)
                shape = list(range(int(num_zeros * (1-missing_ratio)), num_zeros )) 
                index = torch.tensor(shape).to(device)

                if index.numel() > 0:
                    X_1_blank = X_1.index_fill(3, index, 0)
                else:
                    X_1_blank = X_1
                
            else:
                if missing_ratio == 0:
                    X_1_blank = X_1
                else:
                    if args.dataset == 'MIMIC':
                        missing_ratio_to_item_A = {
                            0.1: 3,
                            0.3: 4,
                            0.5: 5,
                            0.7: 6,
                            0.9: 7,
                            1.0: 8
                        }
                        Missing_item_A = missing_ratio_to_item_A.get(missing_ratio, None)  
                        
                        X_1[:,-Missing_item_A:] = 0
                        X_1_blank = X_1
                       
                    elif args.dataset == 'bank':
                        missing_ratio_to_item_A = {
                            0.1: 1,
                            0.3: 2,
                            0.5: 3,
                            0.7: 4,
                            0.9: 5,
                            1.0: 29
                        }
                        Missing_item_A = missing_ratio_to_item_A.get(missing_ratio, None)  
                        
                        X_1[:,-Missing_item_A:] = 0
                        X_1_blank = X_1
                        
                    elif args.dataset == 'avazu':
                        missing_ratio_to_item_A = {
                            0.1: 2,
                            0.3: 3,
                            0.5: 4,
                            0.7: 14,
                            0.9: 24,
                            1.0: 39
                        }
                        Missing_item_A = missing_ratio_to_item_A.get(missing_ratio, None)  

                        X_1[:,-Missing_item_A:] = 0
                        X_1_blank = X_1
            
            VFL_fx1_blank = VFL_Client1_model(X_1_blank)
            _, VFL_server_basedB = VFL_Server_model(VFL_fx1_blank, VFL_fx2)
            VFL_predicted_class_basedB = VFL_server_basedB.argmax(1)
            
            IAVFL_1_fx1_blank = IAVFL_1_Client1_model(X_1_blank)
            _, IAVFL_1_basedB = IAVFL_1_Server_model(IAVFL_1_fx1_blank, IAVFL_1_fx1_blank)
            IAVFL_1_predicted_class_basedB = IAVFL_1_basedB.argmax(1)
            
            predictions_basedB = torch.stack([VFL_predicted_class_basedB, IAVFL_1_predicted_class_basedB, IAVFL_2_predicted_class], dim=0)  # Shape: (3, batch_size)
            final_predictions_basedB, _ = torch.mode(predictions_basedB, dim=0) 
            
            correct_basedB[i] += (final_predictions_basedB == Y_1).type(torch.float).sum().item()
        

        if batch_id == len(test_data) - 1:
            print("---Normal Result (No Missing)---", file=filename)
            print(correct_normal, file=filename)
            correct_normal_ratio = correct_normal/size
            print(f"Vote acc: {(100 * correct_normal_ratio):>0.3f}%", file=filename)
            
            missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
            for i, missing_ratio in enumerate(missing_ratios):
                print(correct_basedA[i], file=filename)
                print("\n", file=filename)
                print(f"***Missing_ratio***: {missing_ratio}", file=filename)
                #print("\n", file=filename)
                print("=======Average Vote Acc================", file=filename)
                average_correct_ratio = (correct_basedA[i] + correct_basedB[i])/(2*size)
                print(f"Average Vote Acc: {(100 * average_correct_ratio):>0.3f}%", file=filename)
                #print("\n", file=filename)
                print("=======A is completed================", file=filename)
                correct_ratio_basedA = correct_basedA[i]/size
                print(f"BasedA Vote Acc: {(100 * correct_ratio_basedA):>0.3f}%", file=filename)
                #print("\n", file=filename)
                print("=======B is completed================", file=filename)
                correct_ratio_basedB = correct_basedB[i]/size
                print(f"BasedB Vote Acc: {(100 * correct_ratio_basedB):>0.3f}%", file=filename)