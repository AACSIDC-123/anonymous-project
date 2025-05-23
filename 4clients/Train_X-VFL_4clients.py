import os
from torch import nn
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import *
from utils import *
from parse import args
import matplotlib.pyplot as plt
from Decoder import *
import torch.nn.functional as F

# Define a random_seed
set_random_seed(args.seed)
MISSING_RATIO_NUM_CIFAR10 = 6
MISSING_RATIO_NUM_MIMIC = 4
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
device = torch.device(f"cuda") 

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

def train_client(train_data, client_model_1, client_model_2, client_model_3, client_model_4,\
            server_model, xCOM_1, xCOM_2, xCOM_3, xCOM_4, t):

    client_model_1.train()
    client_model_2.train()
    client_model_3.train()
    client_model_4.train()
    server_model.train()
    xCOM_1.train()
    xCOM_2.train()
    xCOM_3.train()
    xCOM_4.train()
    
    correct_overlap= 0
    correct_complete1= 0
    correct_complete2= 0
    correct_complete3= 0
    correct_complete4= 0

    loss_overlap = 0
    loss_complete1 = 0
    loss_complete2 = 0
    loss_complete3 = 0
    loss_complete4 = 0
    
    size = len(train_data)*args.batch_size

    for batch_id, batch in enumerate(train_data):
        if args.dataset =='cifar10':
            missing_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
            Missing_ratio = missing_ratios[(batch_id+t) % MISSING_RATIO_NUM_CIFAR10]
        
        elif args.dataset == 'MIMIC':
            missing_ratios = [0.3, 0.5, 0.7, 1]
            Missing_ratio = missing_ratios[(batch_id+t) % MISSING_RATIO_NUM_MIMIC]
        
        X, target = batch
        X_1, X_2, X_3, X_4 = split_data_4clients(args.dataset, X)
        
        if args.dataset == 'utkface' or args.dataset == 'celeba':
            Y_1 = target[0].to(device)
            Y_2 = target[1].view(-1, 1).to(device)
        else:
            Y_1 = target.to(device)
        
        # x_overlap & non-overlap
        X_1_overlap   = X_1[:args.num_overlap]
        X_1_complete1 = X_1[args.num_overlap : int(((args.batch_size - args.num_overlap)/4)*1 + args.num_overlap)]
        X_1_complete2 = X_1[int(((args.batch_size - args.num_overlap)/4)*1 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*2 + args.num_overlap)]
        X_1_complete3 = X_1[int(((args.batch_size - args.num_overlap)/4)*2 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*3 + args.num_overlap)]
        X_1_complete4 = X_1[int(((args.batch_size - args.num_overlap)/4)*3 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*4 + args.num_overlap)]

        X_2_overlap   = X_2[:args.num_overlap]
        X_2_complete1 = X_2[args.num_overlap : int(((args.batch_size - args.num_overlap)/4)*1 + args.num_overlap)]
        X_2_complete2 = X_2[int(((args.batch_size - args.num_overlap)/4)*1 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*2 + args.num_overlap)]
        X_2_complete3 = X_2[int(((args.batch_size - args.num_overlap)/4)*2 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*3 + args.num_overlap)]
        X_2_complete4 = X_2[int(((args.batch_size - args.num_overlap)/4)*3 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*4 + args.num_overlap)]

        X_3_overlap   = X_3[:args.num_overlap]
        X_3_complete1 = X_3[args.num_overlap : int(((args.batch_size - args.num_overlap)/4)*1 + args.num_overlap)]
        X_3_complete2 = X_3[int(((args.batch_size - args.num_overlap)/4)*1 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*2 + args.num_overlap)]
        X_3_complete3 = X_3[int(((args.batch_size - args.num_overlap)/4)*2 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*3 + args.num_overlap)]
        X_3_complete4 = X_3[int(((args.batch_size - args.num_overlap)/4)*3 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*4 + args.num_overlap)]

        X_4_overlap   = X_4[:args.num_overlap]
        X_4_complete1 = X_4[args.num_overlap : int(((args.batch_size - args.num_overlap)/4)*1 + args.num_overlap)]
        X_4_complete2 = X_4[int(((args.batch_size - args.num_overlap)/4)*1 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*2 + args.num_overlap)]
        X_4_complete3 = X_4[int(((args.batch_size - args.num_overlap)/4)*2 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*3 + args.num_overlap)]
        X_4_complete4 = X_4[int(((args.batch_size - args.num_overlap)/4)*3 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*4 + args.num_overlap)]

        Y_overlap   = Y_1[:args.num_overlap]
        Y_complete1 = Y_1[args.num_overlap : int(((args.batch_size - args.num_overlap)/4)*1 + args.num_overlap)]
        Y_complete2 = Y_1[int(((args.batch_size - args.num_overlap)/4)*1 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*2 + args.num_overlap)]
        Y_complete3 = Y_1[int(((args.batch_size - args.num_overlap)/4)*2 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*3 + args.num_overlap)]
        Y_complete4 = Y_1[int(((args.batch_size - args.num_overlap)/4)*3 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*4 + args.num_overlap)]
        
        num_complete1 = int((args.batch_size - args.num_overlap)/4)
        num_complete2 = int((args.batch_size - args.num_overlap)/4)
        num_complete3 = int((args.batch_size - args.num_overlap)/4)
        num_complete4 = int((args.batch_size - args.num_overlap)/4)

        if args.overlap == 'True':
            #E
            fx1 = client_model_1(X_1_overlap)
            fx2 = client_model_2(X_2_overlap)
            fx3 = client_model_3(X_3_overlap)
            fx4 = client_model_4(X_4_overlap)
            
            _, server_fx1 = server_model(fx1, fx1, fx1, fx1)
            _, server_fx2 = server_model(fx2, fx2, fx2, fx2)
            _, server_fx3 = server_model(fx3, fx3, fx3, fx3)
            _, server_fx4 = server_model(fx4, fx4, fx4, fx4)
            
            _, server_fx_1234_avg = server_model(fx1, fx2, fx3, fx4)
    
            # xCOM
            xCOM_1_input = (fx2 + fx3 + fx4)/3
            xCOM_2_input = (fx1 + fx3 + fx4)/3
            xCOM_3_input = (fx1 + fx2 + fx4)/3
            xCOM_4_input = (fx1 + fx2 + fx3)/3
            
            if args.dataset == 'cifar10':
                fx1_p = xCOM_1(xCOM_1_input, X_1_overlap, (1 - Missing_ratio), args.num_overlap, 1)
                fx2_p = xCOM_2(xCOM_2_input, X_2_overlap, (1 - Missing_ratio), args.num_overlap, 2)
                fx3_p = xCOM_3(xCOM_3_input, X_3_overlap, (1 - Missing_ratio), args.num_overlap, 3)
                fx4_p = xCOM_4(xCOM_4_input, X_4_overlap, (1 - Missing_ratio), args.num_overlap, 4)
            elif args.dataset == 'MIMIC':
                Missing_item_1, Missing_item_2, Missing_item_3, Missing_item_4 = get_Missing_item_MIMIC(Missing_ratio)
                fx1_p = xCOM_1(xCOM_1_input, X_1_overlap, Missing_item_1, args.num_overlap)
                fx2_p = xCOM_2(xCOM_2_input, X_2_overlap, Missing_item_2, args.num_overlap)
                fx3_p = xCOM_3(xCOM_3_input, X_3_overlap, Missing_item_3, args.num_overlap)
                fx4_p = xCOM_4(xCOM_4_input, X_4_overlap, Missing_item_4, args.num_overlap)
                
            fx1_xCOM = client_model_1(fx1_p)
            fx2_xCOM = client_model_2(fx2_p)
            fx3_xCOM = client_model_3(fx3_p)
            fx4_xCOM = client_model_4(fx4_p)

            _, server_fx1_xCOM = server_model(fx1_xCOM, fx1_xCOM, fx1_xCOM, fx1_xCOM)
            _, server_fx2_xCOM = server_model(fx2_xCOM, fx2_xCOM, fx2_xCOM, fx2_xCOM)
            _, server_fx3_xCOM = server_model(fx3_xCOM, fx3_xCOM, fx3_xCOM, fx3_xCOM)
            _, server_fx4_xCOM = server_model(fx4_xCOM, fx4_xCOM, fx4_xCOM, fx4_xCOM)

            _, server_fx1_p = server_model(fx1_xCOM, fx2, fx3, fx4)
            _, server_fx2_p = server_model(fx1, fx2_xCOM, fx3, fx4)
            _, server_fx3_p = server_model(fx1, fx2, fx3_xCOM, fx4)
            _, server_fx4_p = server_model(fx1, fx2, fx3, fx4_xCOM)

            L_main = criterion(server_fx1, Y_overlap) + criterion(server_fx2, Y_overlap) + \
                    criterion(server_fx3, Y_overlap) + criterion(server_fx4, Y_overlap) + \
                    criterion(server_fx_1234_avg, Y_overlap) + criterion(server_fx1_p, Y_overlap) + \
                    criterion(server_fx2_p, Y_overlap) + criterion(server_fx3_p, Y_overlap) + \
                    criterion(server_fx4_p, Y_overlap)
            
            L_DS_1 = ((server_fx1_xCOM - server_fx1)**2).sum() + ((server_fx2_xCOM - server_fx2)**2).sum() + \
                    ((server_fx3_xCOM - server_fx3)**2).sum() + ((server_fx4_xCOM - server_fx4)**2).sum()
                    
            L_DS_2 = ((server_fx1 - server_fx_1234_avg)**2).sum() + ((server_fx2 - server_fx_1234_avg)**2).sum() + \
                    ((server_fx3 - server_fx_1234_avg)**2).sum() + ((server_fx4 - server_fx_1234_avg)**2).sum()
            
            L_DS = args.lambda1 * (1/server_fx1.shape[0]) * L_DS_1 + args.lambda2 * (1/server_fx1.shape[0]) * L_DS_2
        
            loss = L_main + L_DS 

            optimizer_client1.zero_grad()
            optimizer_client2.zero_grad() 
            optimizer_client3.zero_grad()
            optimizer_client4.zero_grad()
            optimizer_xCOM1.zero_grad() 
            optimizer_xCOM2.zero_grad()
            optimizer_xCOM3.zero_grad()
            optimizer_xCOM4.zero_grad()
            optimizer_server.zero_grad()

            loss.backward()

            optimizer_client1.step()
            optimizer_client2.step() 
            optimizer_client3.step()
            optimizer_client4.step()
            optimizer_xCOM1.step() 
            optimizer_xCOM2.step()
            optimizer_xCOM3.step()
            optimizer_xCOM4.step()
            optimizer_server.step()

            correct_overlap += (server_fx_1234_avg.argmax(1) == Y_overlap).type(torch.float).sum().item()
            loss_overlap += loss.item()
            
            if batch_id == (args.num_batch-1):
                #acc
                correct_train = correct_overlap / (args.num_batch*args.num_overlap)
                loss, current = loss_overlap/args.num_batch, (batch_id + 1) *args.batch_size
                print(f"Train-overlap-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.2f}%", file=filename)
                train_overlap_acc.append(100 * correct_train)
                train_overlap_loss.append(loss)
        
        if args.complete_1 == 'True':
            
            fx2 = client_model_2(X_2_complete1)
            fx3 = client_model_3(X_3_complete1)
            fx4 = client_model_4(X_4_complete1)    
            
            _, server_fx2 = server_model(fx2, fx2, fx2, fx2)
            _, server_fx3 = server_model(fx3, fx3, fx3, fx3)
            _, server_fx4 = server_model(fx4, fx4, fx4, fx4)
            
            avg_input = (fx2 + fx3 + fx4)/3
            _, server_fx_234_avg = server_model(avg_input, avg_input, avg_input, avg_input)
            
            if args.dataset == 'cifar10':
                fx1_p = xCOM_1(avg_input, X_1_complete1, (1 - Missing_ratio), num_complete1, 1)
            elif args.dataset == 'MIMIC':
                Missing_item_1, Missing_item_2, Missing_item_3, Missing_item_4 = get_Missing_item_MIMIC(Missing_ratio)
                fx1_p = xCOM_1(avg_input, X_1_complete1, Missing_item_1, num_complete1)
                
            fx1_xCOM = client_model_1(fx1_p)
            
            _, server_fx_1p234_avg = server_model(fx1_xCOM, fx2, fx3, fx4)
            
            L_main = criterion(server_fx2, Y_complete1) + criterion(server_fx3, Y_complete1) + \
                criterion(server_fx4, Y_complete1) + criterion(server_fx_234_avg, Y_complete1) + \
                criterion(server_fx_1p234_avg, Y_complete1)
            
            loss = L_main
            
            optimizer_client1.zero_grad()
            optimizer_client2.zero_grad() 
            optimizer_client3.zero_grad()
            optimizer_client4.zero_grad()
            optimizer_xCOM1.zero_grad() 
            optimizer_server.zero_grad()

            loss.backward()

            optimizer_client1.step()
            optimizer_client2.step() 
            optimizer_client3.step()
            optimizer_client4.step()
            optimizer_xCOM1.step() 
            optimizer_server.step()
            
            correct_complete1 += (server_fx_234_avg.argmax(1) == Y_complete1).type(torch.float).sum().item()
            loss_complete1 += loss.item()
            
            if batch_id == args.num_batch-1: 

                correct_train = correct_complete1 / (num_complete1 * args.num_batch)
                loss, current = loss_complete1/args.num_batch, (batch_id + 1) *args.batch_size
                print(f"Train-complete1-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.2f}%",
                    file=filename)
                train_complete1_acc.append(100 * correct_train)
                train_complete1_loss.append(loss)     
                
        elif args.complete_1 == 'False' and batch_id == args.num_batch-1:
            train_complete1_acc.append(0)
            train_complete1_loss.append(0)

        if args.complete_2 == 'True':
            
            fx1 = client_model_1(X_1_complete2)
            fx3 = client_model_3(X_3_complete2)
            fx4 = client_model_4(X_4_complete2)    
            
            _, server_fx1 = server_model(fx1, fx1, fx1, fx1)
            _, server_fx3 = server_model(fx3, fx3, fx3, fx3)
            _, server_fx4 = server_model(fx4, fx4, fx4, fx4)
            
            avg_input = (fx1 + fx3 + fx4)/3
            _, server_fx_134_avg = server_model(avg_input, avg_input, avg_input, avg_input)
            
            if args.dataset == 'cifar10':
                fx2_p = xCOM_2(avg_input, X_2_complete2, (1 - Missing_ratio), num_complete2, 2)
            elif args.dataset == 'MIMIC':
                Missing_item_1, Missing_item_2, Missing_item_3, Missing_item_4 = get_Missing_item_MIMIC(Missing_ratio)
                fx2_p = xCOM_2(avg_input, X_2_complete2, Missing_item_2, num_complete2)
                
            fx2_xCOM = client_model_2(fx2_p)
            
            _, server_fx_12p34_avg = server_model(fx1, fx2_xCOM, fx3, fx4)
            
            L_main = criterion(server_fx1, Y_complete2) + criterion(server_fx3, Y_complete2) + \
                criterion(server_fx4, Y_complete2) + criterion(server_fx_134_avg, Y_complete2) + \
                criterion(server_fx_12p34_avg, Y_complete2)
            
            loss = L_main
            
            optimizer_client1.zero_grad()
            optimizer_client2.zero_grad() 
            optimizer_client3.zero_grad()
            optimizer_client4.zero_grad()
            optimizer_xCOM2.zero_grad() 
            optimizer_server.zero_grad()

            loss.backward()

            optimizer_client1.step()
            optimizer_client2.step() 
            optimizer_client3.step()
            optimizer_client4.step()
            optimizer_xCOM2.step() 
            optimizer_server.step()
            
            correct_complete2 += (server_fx_134_avg.argmax(1) == Y_complete2).type(torch.float).sum().item()
            loss_complete2 += loss.item()
            
            if batch_id == args.num_batch-1: 

                correct_train = correct_complete2 / (num_complete2 * args.num_batch)
                loss, current = loss_complete2/args.num_batch, (batch_id + 1) *args.batch_size
                print(f"Train-complete2-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.2f}%",
                    file=filename)
                train_complete2_acc.append(100 * correct_train)
                train_complete2_loss.append(loss)     
                
        elif args.complete_2 == 'False' and batch_id == args.num_batch-1:
            train_complete2_acc.append(0)
            train_complete2_loss.append(0)
        
        if args.complete_3 == 'True':
            
            fx1 = client_model_1(X_1_complete3)
            fx2 = client_model_2(X_2_complete3)
            fx4 = client_model_4(X_4_complete3)    
            
            _, server_fx1 = server_model(fx1, fx1, fx1, fx1)
            _, server_fx2 = server_model(fx2, fx2, fx2, fx2)
            _, server_fx4 = server_model(fx4, fx4, fx4, fx4)
            
            avg_input = (fx1 + fx2 + fx4)/3
            _, server_fx_124_avg = server_model(avg_input, avg_input, avg_input, avg_input)
            
            if args.dataset == 'cifar10':
                fx3_p = xCOM_3(avg_input, X_3_complete3, (1 - Missing_ratio), num_complete3, 3)
            elif args.dataset == 'MIMIC':
                Missing_item_1, Missing_item_2, Missing_item_3, Missing_item_4 = get_Missing_item_MIMIC(Missing_ratio)
                fx3_p = xCOM_3(avg_input, X_3_complete3, Missing_item_3, num_complete3)
                
            fx3_xCOM = client_model_3(fx3_p)
            
            _, server_fx_123p4_avg = server_model(fx1, fx2, fx3_xCOM, fx4)
            
            L_main = criterion(server_fx1, Y_complete3) + criterion(server_fx2, Y_complete3) + \
                criterion(server_fx4, Y_complete3) + criterion(server_fx_124_avg, Y_complete3) + \
                criterion(server_fx_123p4_avg, Y_complete3)
            
            loss = L_main
            
            optimizer_client1.zero_grad()
            optimizer_client2.zero_grad() 
            optimizer_client3.zero_grad()
            optimizer_client4.zero_grad()
            optimizer_xCOM3.zero_grad() 
            optimizer_server.zero_grad()

            loss.backward()

            optimizer_client1.step()
            optimizer_client2.step() 
            optimizer_client3.step()
            optimizer_client4.step()
            optimizer_xCOM3.step() 
            optimizer_server.step()
            
            correct_complete3 += (server_fx_124_avg.argmax(1) == Y_complete3).type(torch.float).sum().item()
            loss_complete3 += loss.item()
            
            if batch_id == args.num_batch-1: 

                correct_train = correct_complete3 / (num_complete3 * args.num_batch)
                loss, current = loss_complete3/args.num_batch, (batch_id + 1) *args.batch_size
                print(f"Train-complete3-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.2f}%",
                    file=filename)
                train_complete3_acc.append(100 * correct_train)
                train_complete3_loss.append(loss)     
                
        elif args.complete_3 == 'False' and batch_id == args.num_batch-1:
            train_complete3_acc.append(0)
            train_complete3_loss.append(0)
            
        if args.complete_4 == 'True':
            
            fx1 = client_model_1(X_1_complete4)
            fx2 = client_model_2(X_2_complete4)
            fx3 = client_model_3(X_3_complete4)

            _, server_fx1 = server_model(fx1, fx1, fx1, fx1)
            _, server_fx2 = server_model(fx2, fx2, fx2, fx2)
            _, server_fx3 = server_model(fx3, fx3, fx3, fx3)
            
            avg_input = (fx1 + fx2 + fx3)/3
            _, server_fx_123_avg = server_model(avg_input, avg_input, avg_input, avg_input)
            
            if args.dataset == 'cifar10':
                fx4_p = xCOM_4(avg_input, X_4_complete4, (1 - Missing_ratio), num_complete4, 4)
            elif args.dataset == 'MIMIC':
                Missing_item_1, Missing_item_2, Missing_item_3, Missing_item_4 = get_Missing_item_MIMIC(Missing_ratio)
                fx4_p = xCOM_4(avg_input, X_4_complete4, Missing_item_4, num_complete4)
                
            fx4_xCOM = client_model_4(fx4_p)
            
            _, server_fx_1234p_avg = server_model(fx1, fx2, fx3, fx4_xCOM)
            
            L_main = criterion(server_fx1, Y_complete4) + criterion(server_fx2, Y_complete4) + \
                criterion(server_fx3, Y_complete4) + criterion(server_fx_123_avg, Y_complete4) + \
                criterion(server_fx_1234p_avg, Y_complete4)
            
            loss = L_main
            
            optimizer_client1.zero_grad()
            optimizer_client2.zero_grad() 
            optimizer_client3.zero_grad()
            optimizer_client4.zero_grad()
            optimizer_xCOM4.zero_grad() 
            optimizer_server.zero_grad()

            loss.backward()

            optimizer_client1.step()
            optimizer_client2.step() 
            optimizer_client3.step()
            optimizer_client4.step()
            optimizer_xCOM4.step() 
            optimizer_server.step()
            
            correct_complete4 += (server_fx_123_avg.argmax(1) == Y_complete4).type(torch.float).sum().item()
            loss_complete4 += loss.item()
            
            if batch_id == args.num_batch-1: 

                correct_train = correct_complete4 / (num_complete4 * args.num_batch)
                loss, current = loss_complete4/args.num_batch, (batch_id + 1) *args.batch_size
                print(f"Train-complete4-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.2f}%",
                    file=filename)
                train_complete4_acc.append(100 * correct_train)
                train_complete4_loss.append(loss)     
                
        elif args.complete_4 == 'False' and batch_id == args.num_batch-1:
            train_complete4_acc.append(0)
            train_complete4_loss.append(0)

        if batch_id == args.num_batch-1:  
            save_path1 = f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/client_model_1.pth'
            save_path2 = f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/client_model_2.pth'
            save_path3 = f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/client_model_3.pth'
            save_path4 = f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/client_model_4.pth'
            torch.save(client_model_1.state_dict(), save_path1)
            torch.save(client_model_2.state_dict(), save_path2)
            torch.save(client_model_3.state_dict(), save_path3)
            torch.save(client_model_4.state_dict(), save_path4)
            save_server_path = f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/server.pth'
            torch.save(server_model.state_dict(), save_server_path)
            
            save_xCOM_path1 = f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/xCOM_1.pth'
            save_xCOM_path2 = f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/xCOM_2.pth'
            save_xCOM_path3 = f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/xCOM_3.pth'
            save_xCOM_path4 = f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/xCOM_4.pth'
            torch.save(xCOM_1.state_dict(), save_xCOM_path1)
            torch.save(xCOM_2.state_dict(), save_xCOM_path2)
            torch.save(xCOM_3.state_dict(), save_xCOM_path3)
            torch.save(xCOM_4.state_dict(), save_xCOM_path4)
            
        # break        
        if batch_id >= args.num_batch-1:
            break
            
def test_client(test_data, client_model_1, client_model_2, client_model_3, client_model_4,\
            server_model, xCOM_1, xCOM_2, xCOM_3, xCOM_4, t):
    
    client_model_1.eval()
    client_model_2.eval()
    client_model_3.eval()
    client_model_4.eval()
    server_model.eval()
    xCOM_1.eval()
    xCOM_2.eval()
    xCOM_3.eval()
    xCOM_4.eval()
    
    if args.dataset == 'cifar10':
        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    elif args.dataset == 'MIMIC':
        missing_ratios = [0, 0.3, 0.5, 0.7, 1]
        
    correct_standalone = [[0]*len(missing_ratios) for _ in range(4)]
    correct_sum = [[0]*len(missing_ratios) for _ in range(4)]
    
    size = len(test_data) * args.batch_size

    for batch_id, batch in enumerate(test_data):
        X, target = batch
        X_1, X_2, X_3, X_4 = split_data_4clients(args.dataset, X)

        if args.dataset == 'utkface' or args.dataset == 'celeba':
            Y_1 = target[0].to(device)
        else:
            Y_1 = target.to(device)
        
        if args.dataset == 'cifar10':
            missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        elif args.dataset == 'MIMIC':
            missing_ratios = [0, 0.3, 0.5, 0.7, 1]
        
        fx_server_standalone = [[0]*len(missing_ratios) for _ in range(4)]
        fx_server_sum = [[0]*len(missing_ratios) for _ in range(4)]
        

        fx1 = client_model_1(X_1)
        fx2 = client_model_2(X_2)
        fx3 = client_model_3(X_3)
        fx4 = client_model_4(X_4)
        clients_embeddings = [fx1, fx2, fx3, fx4]
        
        for i in range(4):
            _, fx_server_sum[i][0] = server_model(clients_embeddings[0], clients_embeddings[1], clients_embeddings[2], clients_embeddings[3])
            correct_sum[i][0] += (fx_server_sum[i][0].argmax(1) == Y_1).type(torch.float).sum().item()
            
            _, fx_server_standalone[i][0] = server_model(clients_embeddings[i], clients_embeddings[i], clients_embeddings[i], clients_embeddings[i])
            correct_standalone[i][0] += (fx_server_standalone[i][0].argmax(1) == Y_1).type(torch.float).sum().item()
            
        for i, missing_ratio in enumerate(missing_ratios):
            if i == 0:
                continue
            else:
                xCOM_1_input = (clients_embeddings[1]+clients_embeddings[2]+clients_embeddings[3])/3
                xCOM_2_input = (clients_embeddings[0]+clients_embeddings[2]+clients_embeddings[3])/3
                xCOM_3_input = (clients_embeddings[0]+clients_embeddings[1]+clients_embeddings[3])/3
                xCOM_4_input = (clients_embeddings[0]+clients_embeddings[1]+clients_embeddings[2])/3
                
                if args.dataset == 'cifar10':
                    X1_xCOM = xCOM_1(xCOM_1_input, X_1, (1 - missing_ratio), args.batch_size, 1)
                    X2_xCOM = xCOM_2(xCOM_2_input, X_2, (1 - missing_ratio), args.batch_size, 2)
                    X3_xCOM = xCOM_3(xCOM_3_input, X_3, (1 - missing_ratio), args.batch_size, 3)
                    X4_xCOM = xCOM_4(xCOM_4_input, X_4, (1 - missing_ratio), args.batch_size, 4)
                elif args.dataset == 'MIMIC':
                    Missing_item_1, Missing_item_2, Missing_item_3, Missing_item_4 = get_Missing_item_MIMIC(missing_ratio)
                    X1_xCOM = xCOM_1(xCOM_1_input, X_1, Missing_item_1, args.batch_size)
                    X2_xCOM = xCOM_2(xCOM_2_input, X_2, Missing_item_2, args.batch_size)
                    X3_xCOM = xCOM_3(xCOM_3_input, X_3, Missing_item_3, args.batch_size)
                    X4_xCOM = xCOM_4(xCOM_4_input, X_4, Missing_item_4, args.batch_size)    
                    
                fx1_xCOM = client_model_1(X1_xCOM)
                fx2_xCOM = client_model_2(X2_xCOM)
                fx3_xCOM = client_model_3(X3_xCOM)
                fx4_xCOM = client_model_4(X4_xCOM)
                
                _, fx_server_sum[0][i] = server_model(fx1_xCOM, clients_embeddings[1], clients_embeddings[2], clients_embeddings[3])
                correct_sum[0][i] += (fx_server_sum[0][i].argmax(1) == Y_1).type(torch.float).sum().item()
                
                _, fx_server_standalone[0][i] = server_model(fx1_xCOM, fx1_xCOM, fx1_xCOM, fx1_xCOM)
                correct_standalone[0][i] += (fx_server_standalone[0][i].argmax(1) == Y_1).type(torch.float).sum().item()
                
                _, fx_server_sum[1][i] = server_model(clients_embeddings[0], fx2_xCOM, clients_embeddings[2], clients_embeddings[3])
                correct_sum[1][i] += (fx_server_sum[1][i].argmax(1) == Y_1).type(torch.float).sum().item()
                
                _, fx_server_standalone[1][i] = server_model(fx2_xCOM, fx2_xCOM, fx2_xCOM, fx2_xCOM)
                correct_standalone[1][i] += (fx_server_standalone[1][i].argmax(1) == Y_1).type(torch.float).sum().item()
                
                _, fx_server_sum[2][i] = server_model(clients_embeddings[0], clients_embeddings[1], fx3_xCOM, clients_embeddings[3])
                correct_sum[2][i] += (fx_server_sum[2][i].argmax(1) == Y_1).type(torch.float).sum().item()
                
                _, fx_server_standalone[2][i] = server_model(fx3_xCOM, fx3_xCOM, fx3_xCOM, fx3_xCOM)
                correct_standalone[2][i] += (fx_server_standalone[2][i].argmax(1) == Y_1).type(torch.float).sum().item()
                
                _, fx_server_sum[3][i] = server_model(clients_embeddings[0], clients_embeddings[1], clients_embeddings[2], fx4_xCOM)
                correct_sum[3][i] += (fx_server_sum[3][i].argmax(1) == Y_1).type(torch.float).sum().item()

                _, fx_server_standalone[3][i] = server_model(fx4_xCOM, fx4_xCOM, fx4_xCOM, fx4_xCOM)
                correct_standalone[3][i] += (fx_server_standalone[3][i].argmax(1) == Y_1).type(torch.float).sum().item()
                
        if batch_id == len(test_data) - 1:
            for i in range(len(correct_standalone)):
                for j in range(len(correct_standalone[0])):
                    test_standalone_acc[t][i][j] = correct_standalone[i][j]/size
                    test_sum_acc[t][i][j] = correct_sum[i][j]/size
                    
            if args.dataset == 'cifar10':
                missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
            elif args.dataset == 'MIMIC':
                missing_ratios = [0, 0.3, 0.5, 0.7, 1]
                
            print("-----------", file=filename)
            for client_id in range(4):
                print(f"\nClient {client_id+1}: ", file=filename)
                for i, missing_ratio in enumerate(missing_ratios):
                    print(f"Missing ratio: {missing_ratio}", file=filename)
                    print(f"Standalone Acc: {(100 *test_standalone_acc[t][client_id][i]):>0.2f}%", file=filename)
                    print(f"Collaborate Acc: {(100 *test_sum_acc[t][client_id][i]):>0.2f}%", file=filename)
                    
def find_top_k_average(input_list, k=5):
    
    # transfer to tensor
    input_tensor = torch.tensor(input_list)

    if args.dataset == 'cifar10':
        # shape: (4, 7)
        topk_avg = torch.zeros((4, 7))
    elif args.dataset == 'MIMIC':
        # shape: (4, 5)
        topk_avg = torch.zeros((4, 5))
        
    for i in range(4):
        if args.dataset == 'cifar10':
            for j in range(7):
                values = input_tensor[:, i, j]           
                topk = torch.topk(values, k).values     
                topk_avg[i][j] = topk.mean()             
        elif args.dataset == 'MIMIC':
            for j in range(5):
                values = input_tensor[:, i, j]           
                topk = torch.topk(values, k).values      
                topk_avg[i][j] = topk.mean()  
    
    # return list of list
    return topk_avg.tolist()

if __name__ == '__main__':
    # Define record path
    root_path = '.'
    data_path = os.path.join(root_path, 'Dataset').replace('\\', '/')
    save_path = f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = open(f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/A_training_process.txt', 'w+')

    ### Load data 
    if args.dataset == 'utkface' or args.dataset == 'celeba':  
        train_data_nondataloader, test_data, num_classes, num_classes2, channel, hidden = load_data(args.dataset, args.attributes, data_path, args.batch_size)
    elif args.dataset == "imagenet":
        train_data_nondataloader, test_data, num_classes, channel, hidden = get_tinyimagenet_bothloader(batch_size=args.batch_size, shuffle=True, seed=args.seed, device=device, classes = args.classes)
    elif args.dataset =='MIMIC':
        train_data_nondataloader, test_data, input_dim1, input_dim2, input_dim3, input_dim4, num_classes = load_data_tabular_4clients(args.dataset, args.batch_size)
    else:
        train_data_nondataloader, test_data, num_classes, channel, hidden = gen_dataset(args.dataset, data_path, device = device)

    if args.dataset =='cifar10' or args.dataset =='imagenet' or args.dataset =='utkface':

        # Define model
        ImageModel_int= ImageModel_4clients(dataset= args.dataset, hidden=hidden, num_cutlayer=args.num_cutlayer, num_classes=num_classes, mode='sum', device=device)
        client_model_1, client_model_2, client_model_3, client_model_4, server_model = ImageModel_int.GetModel()

        # Define decoder
        xCOM_1 = Generator_cifar_4clients(channel=channel, shape_img=32, n_hidden_1=4096, batchsize=args.batch_size, g_in=args.num_cutlayer, iters=1).to(device)
        xCOM_2 = Generator_cifar_4clients(channel=channel, shape_img=32, n_hidden_1=4096, batchsize=args.batch_size, g_in=args.num_cutlayer, iters=2).to(device)
        xCOM_3 = Generator_cifar_4clients(channel=channel, shape_img=32, n_hidden_1=4096, batchsize=args.batch_size, g_in=args.num_cutlayer, iters=3).to(device)
        xCOM_4 = Generator_cifar_4clients(channel=channel, shape_img=32, n_hidden_1=4096, batchsize=args.batch_size, g_in=args.num_cutlayer, iters=4).to(device)
    
    elif args.dataset =='MIMIC':
        
        # Define model 
        client_model_1, client_model_2, client_model_3, client_model_4, server_model = def_tabular_model_4clients(dataset= args.dataset, level= args.level, input_dim1=input_dim1, input_dim2=input_dim2, input_dim3=input_dim3, input_dim4=input_dim4, num_classes=num_classes, num_cutlayer=args.num_cutlayer, mode='sum', device=device)

        # Define decoder 
        if args.level == 3:
            
            xCOM_1 = Generator_mimic_3_4clients(in_dim=input_dim1, n_hidden_1=8, n_hidden_2=32, n_hidden_3=128, n_hidden_4=args.num_cutlayer).to(device)
            xCOM_2 = Generator_mimic_3_4clients(in_dim=input_dim2, n_hidden_1=8, n_hidden_2=32, n_hidden_3=128, n_hidden_4=args.num_cutlayer).to(device)
            xCOM_3 = Generator_mimic_3_4clients(in_dim=input_dim3, n_hidden_1=8, n_hidden_2=32, n_hidden_3=128, n_hidden_4=args.num_cutlayer).to(device)
            xCOM_4 = Generator_mimic_3_4clients(in_dim=input_dim4, n_hidden_1=8, n_hidden_2=32, n_hidden_3=128, n_hidden_4=args.num_cutlayer).to(device)
        else:
            raise ValueError("Only level 3 is supported.")

    # criterion & optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer_client1 = torch.optim.Adam(client_model_1.parameters(), lr=args.lr, foreach=False)
    optimizer_client2 = torch.optim.Adam(client_model_2.parameters(), lr=args.lr, foreach=False)
    optimizer_client3 = torch.optim.Adam(client_model_3.parameters(), lr=args.lr, foreach=False)
    optimizer_client4 = torch.optim.Adam(client_model_4.parameters(), lr=args.lr, foreach=False)
    optimizer_server  = torch.optim.Adam(server_model.parameters(),   lr=args.lr, foreach=False)  

    optimizer_xCOM1 = torch.optim.Adam(xCOM_1.parameters(), lr=args.lr, foreach=False)
    optimizer_xCOM2 = torch.optim.Adam(xCOM_2.parameters(), lr=args.lr, foreach=False)
    optimizer_xCOM3 = torch.optim.Adam(xCOM_3.parameters(), lr=args.lr, foreach=False)
    optimizer_xCOM4 = torch.optim.Adam(xCOM_4.parameters(), lr=args.lr, foreach=False)

    train_overlap_loss = []
    train_overlap_acc = []
    train_complete1_loss = []
    train_complete1_acc = []
    train_complete2_loss = []
    train_complete2_acc = []
    train_complete3_loss = []
    train_complete3_acc = []
    train_complete4_loss = []
    train_complete4_acc = []
    
    if args.dataset =='cifar10':
        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    elif args.dataset =='MIMIC':
        missing_ratios = [0, 0.3, 0.5, 0.7, 1]
    
    test_standalone_acc = [
        [[0]*len(missing_ratios) for _ in range(4)] 
        for _ in range(args.epochs)
    ]
    
    test_sum_acc = [
        [[0]*len(missing_ratios) for _ in range(4)] 
        for _ in range(args.epochs)
    ]

    # start training
    for t in range(args.epochs):

        if args.dataset == 'cifar10' or args.dataset == "imagenet" or args.dataset == 'utkface' or args.dataset =='MIMIC' or args.dataset == 'bank' or args.dataset == 'avazu':
            train_data = get_train(train_data_nondataloader, args.batch_size, args.seed, device) 
        
        print(f"Epoch {t + 1}\n-------------------------------", file=filename)
        
        train_client(train_data, client_model_1, client_model_2, client_model_3, client_model_4,\
            server_model, xCOM_1, xCOM_2, xCOM_3, xCOM_4, t)
        test_client(test_data, client_model_1, client_model_2, client_model_3, client_model_4,\
            server_model, xCOM_1, xCOM_2, xCOM_3, xCOM_4, t)
        
    print("Done!\n", file=filename)

    k = 5
    test_standalone_acc_topk = find_top_k_average(test_standalone_acc, k = 5)
    test_sum_acc_topk = find_top_k_average(test_sum_acc, k = 5)
    test_standalone_acc_topk_avg = [0]*len(missing_ratios)
    test_sum_acc_topk_avg = [0]*len(missing_ratios)
    print("----Final result----", file=filename)
    print(f"----Find the top {k} result----", file=filename)
    
    if args.dataset == 'cifar10':
        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    elif args.dataset == 'MIMIC':
        missing_ratios = [0, 0.3, 0.5, 0.7, 1]
    
    for i, missing_ratio in enumerate(missing_ratios):
        print(f"\nMissing ratio: {missing_ratio}: ", file=filename)
        print(f"\n-----Case Standalone-----", file=filename)
        for client_id in range(4):
            print(f"Client: {client_id+1} Acc: {(100*test_standalone_acc_topk[client_id][i]):>0.2f}%", file=filename)
        
        test_standalone_acc_topk_avg[i] = (test_standalone_acc_topk[0][i]+test_standalone_acc_topk[1][i]+test_standalone_acc_topk[2][i]+test_standalone_acc_topk[3][i])/4
        print(f"Avg-Acc: {(100*test_standalone_acc_topk_avg[i]):>0.2f}%", file=filename)
        
        print(f"\n-----Case Collaborate-----", file=filename)
        for client_id in range(4):
            print(f"Client: {client_id+1} Acc: {(100*test_sum_acc_topk[client_id][i]):>0.2f}%", file=filename)
            
        test_sum_acc_topk_avg[i] = (test_sum_acc_topk[0][i]+test_sum_acc_topk[1][i]+test_sum_acc_topk[2][i]+test_sum_acc_topk[3][i])/4
        print(f"Avg-Acc: {(100*test_sum_acc_topk_avg[i]):>0.2f}%", file=filename)
        
    #plt
    x = np.arange(len(train_overlap_acc))
    plt.plot(x, train_overlap_acc, label='train_overlap_acc')
    plt.plot(x, train_complete1_acc, label='train_complete1_acc')
    plt.plot(x, train_complete2_acc, label='train_complete2_acc')
    plt.plot(x, train_complete3_acc, label='train_complete3_acc')
    plt.plot(x, train_complete4_acc, label='train_complete4_acc')

    plt.xlabel('epoch', fontsize=19)
    plt.ylabel('ACC', fontsize=19)
    plt.title(f'{args.dataset}_overlap {100*(args.num_overlap/args.batch_size)}%', fontsize=20)
    plt.legend()
    plt.savefig(f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/A_train_ACC.png')
    plt.close()

    x = np.arange(len(train_overlap_loss))
    plt.plot(x, train_overlap_loss, label='train_overlap_loss')
    plt.plot(x, train_complete1_loss, label='train_complete1_loss')
    plt.plot(x, train_complete2_loss, label='train_complete2_loss')
    plt.plot(x, train_complete3_loss, label='train_complete3_loss')
    plt.plot(x, train_complete4_loss, label='train_complete4_loss')

    plt.xlabel('epoch', fontsize=19)
    plt.ylabel('Loss', fontsize=19)
    plt.title(f'{args.dataset}_overlap {100*(args.num_overlap/args.batch_size)}%', fontsize=20)
    plt.legend()
    plt.savefig(f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/A_train_Loss.png')
    plt.close()
    
    tensor_standalone_acc = torch.tensor(test_standalone_acc)             
    mean_standalone_acc = tensor_standalone_acc.mean(dim=1, keepdim=True)             
    mean_standalone_acc_list = mean_standalone_acc.squeeze(1).tolist()  
    
    tensor_sum_acc = torch.tensor(test_sum_acc)             
    mean_sum_acc = tensor_sum_acc.mean(dim=1, keepdim=True)             
    mean_sum_acc_list = mean_sum_acc.squeeze(1).tolist() 
    
    x = np.arange(args.epochs)

    if args.dataset == 'cifar10':
        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    elif args.dataset == 'MIMIC':
        missing_ratios = [0, 0.3, 0.5, 0.7, 1]
        
    for i, missing_ratio in enumerate(missing_ratios):
        y = [row[i] for row in mean_standalone_acc_list]  
        plt.plot(x, y, label=f"Missing ratio: {missing_ratio}")
        
    plt.xlabel('epoch', fontsize=19)
    plt.ylabel('Acc', fontsize=19)
    plt.title(f'{args.dataset}_overlap {100*(args.num_overlap/args.batch_size)}%_standalone_test', fontsize=20)
    plt.legend()
    plt.savefig(f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/A_test_standalone.png')
    plt.close()

    x = np.arange(args.epochs)
    
    if args.dataset == 'cifar10':
        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    elif args.dataset == 'MIMIC':
        missing_ratios = [0, 0.3, 0.5, 0.7, 1]
        
    for i, missing_ratio in enumerate(missing_ratios):
        y = [row[i] for row in mean_sum_acc_list]
        plt.plot(x, y, label=f"Missing ratio: {missing_ratio}")

    plt.xlabel('epoch', fontsize=19)
    plt.ylabel('Acc', fontsize=19)
    plt.title(f'{args.dataset}_overlap {100*(args.num_overlap/args.batch_size)}%_sum_test', fontsize=20)
    plt.legend()
    plt.savefig(f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/A_test_sum.png')
    plt.close()
    
    # save all the results of training & testing
    np.save(f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/train_overlap_acc.npy', train_overlap_acc) 
    np.save(f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/train_complete1_acc.npy', train_complete1_acc) 
    np.save(f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/train_complete2_acc.npy', train_complete2_acc) 
    np.save(f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/train_complete3_acc.npy', train_complete3_acc) 
    np.save(f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/train_complete4_acc.npy', train_complete4_acc) 

    np.save(f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/test_standalone_acc.npy', test_standalone_acc) 
    np.save(f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/test_sum_acc.npy', test_sum_acc) 
    np.save(f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/test_standalone_acc_topk.npy', test_standalone_acc_topk) 
    np.save(f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/test_sum_acc_topk.npy', test_sum_acc_topk) 
    np.save(f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/test_standalone_acc_topk_avg.npy', test_standalone_acc_topk_avg) 
    np.save(f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/test_sum_acc_topk_avg.npy', test_sum_acc_topk_avg) 
    
    np.save(f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/mean_standalone_acc_list.npy', mean_standalone_acc_list) 
    np.save(f'Result/Results_X-VFL_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/mean_sum_acc_list.npy', mean_sum_acc_list) 
    