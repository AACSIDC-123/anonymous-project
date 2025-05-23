import os
from torch import nn
import numpy as np
from models import *
from utils import *
from parse import args
import matplotlib.pyplot as plt
from Decoder import *
import torch.nn.functional as F

# Define a random_seed
set_random_seed(args.seed)
MISSING_RATIO_NUM = 6
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
device = torch.device(f"cuda") 

def train_client(dataloader, client_model_1, client_model_2, t):

    client_model_1.train()
    client_model_2.train()
    server_model.train()
    Decoder12.train()
    Decoder21.train()
    
    correct_overlap= 0
    correct_complete1= 0
    correct_complete2= 0

    size = len(dataloader)*args.batch_size

    for batch_id, batch in enumerate(dataloader):

        missing_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
        Missing_ratio = missing_ratios[batch_id % MISSING_RATIO_NUM]
        
        X, target = batch
        X_1, X_2 = split_data(args.dataset, X)
        
        if args.dataset == 'utkface' or args.dataset == 'celeba':
            Y_1 = target[0].to(device)
            Y_2 = target[1].view(-1, 1).to(device)
        else:
            Y_1 = target.to(device)
        
        # x_overlap & non-overlap
        X_1_overlap   = X_1[:args.num_overlap]
        X_2_overlap   = X_2[:args.num_overlap]

        X_1_complete1 = X_1[args.num_overlap:int((args.batch_size - args.num_overlap)*args.nonoverlap1_ratio + args.num_overlap)]
        X_2_complete1 = X_2[args.num_overlap:int((args.batch_size - args.num_overlap)*args.nonoverlap1_ratio + args.num_overlap)]
        
        X_1_complete2 = X_1[int((args.batch_size - args.num_overlap)*(args.nonoverlap1_ratio) + args.num_overlap):args.batch_size]
        X_2_complete2 = X_2[int((args.batch_size - args.num_overlap)*(args.nonoverlap1_ratio) + args.num_overlap):args.batch_size]
        
        Y_overlap   = Y_1[:args.num_overlap]
        Y_complete1 = Y_1[args.num_overlap:int((args.batch_size - args.num_overlap)*args.nonoverlap1_ratio + args.num_overlap)]
        Y_complete2 = Y_1[int((args.batch_size - args.num_overlap)*(args.nonoverlap1_ratio) + args.num_overlap):args.batch_size]

        num_complete1 = int((args.batch_size - args.num_overlap)*args.nonoverlap1_ratio)
        num_complete2 = args.batch_size - args.num_overlap - num_complete1

        if args.overlap == 'True':
            
            fx1 = client_model_1(X_1_overlap)
            fx2 = client_model_2(X_2_overlap)
            
            _, server_fx1 = server_model(fx1, fx1)
            _, server_fx2 = server_model(fx2, fx2)
            
            _, server_fx_avg = server_model(fx1, fx2)
    
            if args.dataset == 'cifar10' or args.dataset == "imagenet" or args.dataset == 'utkface':
                D12 = Decoder12(fx1, X_2_overlap, (1 - Missing_ratio), args.num_overlap, 1)
                D21 = Decoder21(fx2, X_1_overlap, (1 - Missing_ratio), args.num_overlap, 0)
            elif args.dataset == 'MIMIC':
                missing_ratio_to_item_A = {
                    0.1: 3,
                    0.3: 4,
                    0.5: 5,
                    0.7: 6,
                    0.9: 7,
                    1.0: 8
                }
                Missing_item_A = missing_ratio_to_item_A.get(Missing_ratio, None)  
                missing_ratio_to_item_B = {
                    0.1: 2,
                    0.3: 3,
                    0.5: 4,
                    0.7: 5,
                    0.9: 6,
                    1.0: 7
                }
                Missing_item_B = missing_ratio_to_item_B.get(Missing_ratio, None)  
                
                D12 = Decoder12(fx1, X_2_overlap, Missing_item_B, args.num_overlap)
                D21 = Decoder21(fx2, X_1_overlap, Missing_item_A, args.num_overlap)
                
            elif args.dataset == 'bank':
                missing_ratio_to_item_A = {
                    0.1: 1,
                    0.3: 2,
                    0.5: 3,
                    0.7: 4,
                    0.9: 5,
                    1.0: 29
                }
                Missing_item_A = missing_ratio_to_item_A.get(Missing_ratio, None)  
                missing_ratio_to_item_B = {
                    0.1: 1,
                    0.3: 2,
                    0.5: 3,
                    0.7: 4,
                    0.9: 5,
                    1.0: 28
                }
                Missing_item_B = missing_ratio_to_item_B.get(Missing_ratio, None) 
                
                D12 = Decoder12(fx1, X_2_overlap, Missing_item_B, args.num_overlap)
                D21 = Decoder21(fx2, X_1_overlap, Missing_item_A, args.num_overlap)
                
            elif args.dataset == 'avazu':
                missing_ratio_to_item_A = {
                    0.1: 2,
                    0.3: 3,
                    0.5: 4,
                    0.7: 14,
                    0.9: 24,
                    1.0: 39
                }
                Missing_item_A = missing_ratio_to_item_A.get(Missing_ratio, None)  
                missing_ratio_to_item_B = {
                    0.1: 1,
                    0.3: 2,
                    0.5: 3,
                    0.7: 13,
                    0.9: 23,
                    1.0: 38
                }
                Missing_item_B = missing_ratio_to_item_B.get(Missing_ratio, None)  
                
                D12 = Decoder12(fx1, X_2_overlap, Missing_item_B, args.num_overlap)
                D21 = Decoder21(fx2, X_1_overlap, Missing_item_A, args.num_overlap)
                
            fx21_d = client_model_1(D21)
            fx12_d = client_model_2(D12)

            _, server_fx21 = server_model(fx21_d, fx21_d)
            _, server_fx12 = server_model(fx12_d, fx12_d)

            _, server_fx_comp1 = server_model(fx2, fx21_d)
            _, server_fx_comp2 = server_model(fx1, fx12_d)

            loss_ce_main = criterion(server_fx1, Y_overlap) +  criterion(server_fx2, Y_overlap) +  criterion(server_fx_avg, Y_overlap) \
                        + criterion(server_fx_comp1, Y_overlap) + criterion(server_fx_comp2, Y_overlap)

            loss1 = ((server_fx_avg - server_fx1)**2).sum() + ((server_fx_avg - server_fx2)**2).sum()
            loss2 = ((server_fx1 - server_fx21)**2).sum() + ((server_fx2 - server_fx12)**2).sum()
            loss_reg = (args.lambda1 * (1/server_fx1.shape[0]) * loss1 + 
                        args.lambda2 * (1/server_fx1.shape[0]) * loss2   
                        )
            
            loss = loss_ce_main + loss_reg 

            optimizer_decoder12.zero_grad()
            optimizer_decoder21.zero_grad()
            optimizer_client1.zero_grad()
            optimizer_client2.zero_grad() 
            optimizer_server.zero_grad()

            loss.backward()

            optimizer_client1.step()
            optimizer_client2.step()
            optimizer_decoder12.step()
            optimizer_decoder21.step()
            optimizer_server.step()

            
            correct_overlap += (server_fx_avg.argmax(1) == Y_overlap).type(torch.float).sum().item()
            if batch_id == (args.num_batch-1):
                save_path3 = f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/server.pth'
                torch.save(server_model, save_path3)
                correct_train = correct_overlap / (args.num_batch*args.num_overlap)
                loss, current = loss.item(), (batch_id + 1) *args.batch_size
                print(f"Train-overlap-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%", file=filename)
                train_overlap_acc.append(100 * correct_train)
                train_overlap_loss.append(loss)
        
        if args.complete_1 == 'True':
            #E
            fx2 = client_model_2(X_2_complete1)    
            server_fx2_, server_fx2 = server_model(fx2, fx2)            

            if args.dataset == 'cifar10' or args.dataset == "imagenet" or args.dataset == 'utkface':
                D21 = Decoder21(fx2, X_1_complete1, (1 - Missing_ratio), num_complete1, 0)
                
            elif args.dataset == 'MIMIC':
                missing_ratio_to_item_A = {
                    0.1: 3,
                    0.3: 4,
                    0.5: 5,
                    0.7: 6,
                    0.9: 7,
                    1.0: 8
                }
                Missing_item_A = missing_ratio_to_item_A.get(Missing_ratio, None)  
                D21 = Decoder21(fx2, X_1_complete1, Missing_item_A, num_complete1)  
                
            elif args.dataset == 'bank':
                missing_ratio_to_item_A = {
                    0.1: 1,
                    0.3: 2,
                    0.5: 3,
                    0.7: 4,
                    0.9: 5,
                    1.0: 29
                }
                Missing_item_A = missing_ratio_to_item_A.get(Missing_ratio, None)  
                D21 = Decoder21(fx2, X_1_complete1, Missing_item_A, num_complete1)
                
            elif args.dataset == 'avazu':
                missing_ratio_to_item_A = {
                    0.1: 2,
                    0.3: 3,
                    0.5: 4,
                    0.7: 14,
                    0.9: 24,
                    1.0: 39
                }
                Missing_item_A = missing_ratio_to_item_A.get(Missing_ratio, None)  

                D21 = Decoder21(fx2, X_1_complete1, Missing_item_A, num_complete1)

            fx21_d = client_model_1(D21)
            
            _, server_fx21 = server_model(fx21_d, fx21_d)
          
            _, server_fx_comp1 = server_model(fx2, fx21_d)
        
            loss_ce  = criterion(server_fx2, Y_complete1)        
            loss_reg = criterion(server_fx_comp1, Y_complete1)
            loss = loss_ce  + loss_reg 
            
            optimizer_decoder21.zero_grad()
            optimizer_client1.zero_grad()
            optimizer_client2.zero_grad() 
            optimizer_server.zero_grad()

            loss.backward()

            optimizer_client1.step()
            optimizer_client2.step()
            optimizer_decoder21.step()
            optimizer_server.step()
            
            #acc
            correct_complete1 += (server_fx2.argmax(1) == Y_complete1).type(torch.float).sum().item()

            if batch_id == args.num_batch-1: 

                save_path3 = f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/server.pth'
                torch.save(server_model, save_path3)
                correct_train = correct_complete1 / (num_complete1 * args.num_batch)
                loss, current = loss.item(), (batch_id + 1) *args.batch_size
                print(f"Train-compelet1-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%",
                    file=filename)
                train_complete1_acc.append(100 * correct_train)
                train_complete1_loss.append(loss)     
                
        elif args.complete_1 == 'False' and batch_id == args.num_batch-1:
            train_complete1_acc.append(0)
            train_complete1_loss.append(0)
                
        if args.complete_2 == 'True':
            
            fx1 = client_model_1(X_1_complete2)  
            server_fx1_, server_fx1 = server_model(fx1, fx1)          

            if args.dataset == 'cifar10' or args.dataset == "imagenet" or args.dataset == 'utkface':
                D12 = Decoder12(fx1, X_2_complete2, (1 - Missing_ratio), num_complete2, 1)
                
            elif args.dataset == 'MIMIC':
                missing_ratio_to_item_B = {
                    0.1: 2,
                    0.3: 3,
                    0.5: 4,
                    0.7: 5,
                    0.9: 6,
                    1.0: 7
                }
                Missing_item_B = missing_ratio_to_item_B.get(Missing_ratio, None)  
                D12 = Decoder12(fx1, X_2_complete2, Missing_item_B, num_complete2)
                
            elif args.dataset == 'bank':

                missing_ratio_to_item_B = {
                    0.1: 1,
                    0.3: 2,
                    0.5: 3,
                    0.7: 4,
                    0.9: 5,
                    1.0: 28
                }
                Missing_item_B = missing_ratio_to_item_B.get(Missing_ratio, None) 
                D12 = Decoder12(fx1, X_2_complete2, Missing_item_B, num_complete2)
                
            elif args.dataset == 'avazu':
                
                missing_ratio_to_item_B = {
                    0.1: 1,
                    0.3: 2,
                    0.5: 3,
                    0.7: 13,
                    0.9: 23,
                    1.0: 38
                }
                Missing_item_B = missing_ratio_to_item_B.get(Missing_ratio, None) 
                D12 = Decoder12(fx1, X_2_complete2, Missing_item_B, num_complete2)

            fx12_d = client_model_2(D12)

            _, server_fx12 = server_model(fx12_d, fx12_d)
            _, server_fx_comp2 = server_model(fx1, fx12_d)
            
            loss_ce_main = criterion(server_fx1, Y_complete2) 
            loss_reg = criterion(server_fx_comp2, Y_complete2)
                        
            loss = loss_ce_main + loss_reg  

            optimizer_decoder12.zero_grad()
            optimizer_client1.zero_grad()
            optimizer_client2.zero_grad() 
            optimizer_server.zero_grad()
            
            loss.backward()

            optimizer_client1.step()
            optimizer_client2.step()
            optimizer_decoder12.step()
            optimizer_server.step()

            correct_complete2 += (server_fx1.argmax(1) == Y_complete2).type(torch.float).sum().item()

            if batch_id == args.num_batch-1:
                
                save_path3 = f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/server.pth'
                torch.save(server_model, save_path3)
               
                correct_train = correct_complete2 / (num_complete2 * args.num_batch)
                loss, current = loss.item(), (batch_id + 1) *args.batch_size
                print(f"train-compelet2-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%",
                    file=filename)
                train_complete2_acc.append(100 * correct_train)
                train_complete2_loss.append(loss)

        elif args.complete_2 == 'False':
            train_complete2_acc.append(0)
            train_complete2_loss.append(0)

        if batch_id == args.num_batch-1:
            test_client(test_data, t)
            test_alone1_client(test_data, t)
            test_alone2_client(test_data, t)    
            test_partyA_seperate(test_data, t)
            test_partyB_seperate(test_data, t)

            test_two_partys_decoder(test_data, t)

        if (t+1) %10 ==0 or t == args.epochs-1:
            if batch_id == args.num_batch-1:  
                save_path1 = f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/client1_epoch{t+1}.pth'
                save_path2 = f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/client2_epoch{t+1}.pth'
                torch.save(client_model_1, save_path1)
                torch.save(client_model_2, save_path2)
                save_path3 = f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/server.pth'
                torch.save(server_model, save_path3)
              
        if batch_id >= args.num_batch-1:
            break
            
def test_server(client1_fx, client2_fx, y, batch_id, correct, size):
    server_model.eval()
    correct = correct

    optimizer_server.zero_grad()
    _, fx_server = server_model(client1_fx, client2_fx)
    loss = criterion(fx_server, y)

    correct += (fx_server.argmax(1) == y).type(torch.float).sum().item()
    correct_train = correct / size
    loss, current = loss.item(), (batch_id + 1) * len(y)
    if batch_id == len(test_data) - 1:
        print("Using two partys' all aligned data", file=filename)
        print(f"Test-loss: {loss:>7f}  [{current:>5d}/{size:>5d}], Accuracy: {(100 * correct_train):>0.1f}%",
              file=filename)
        test_acc.append(100 * correct_train)
        test_loss.append(loss)
    return correct

def test_alone1_server(client1_fx, y, batch_id, correct, size, t):
    
    save_path3 = f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/server.pth'
    server_test_model  = torch.load(save_path3, map_location=device)
    server_test_model.eval()
    
    correct = correct

    _, fx_server = server_test_model(client1_fx, client1_fx)
    loss = criterion(fx_server, y)

    correct += (fx_server.argmax(1) == y).type(torch.float).sum().item()
    correct_train = correct / size
    loss, current = loss.item(), (batch_id + 1) * len(y)
    if batch_id == len(test_data) - 1:
        print("Using party' A all data", file=filename)
        print(f"Test-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%",
              file=filename)
        test_alone1_acc.append(100 * correct_train)
        test_alone1_loss.append(loss)
    return correct

def test_alone2_server(client2_fx, y, batch_id, correct, size, t):    
    save_path3 = f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/server.pth'
    server_test_model  = torch.load(save_path3, map_location=device)
    server_test_model.eval()
    correct = correct
    
    # train and update
    _, fx_server = server_test_model(client2_fx, client2_fx)
    loss = criterion(fx_server, y)

    correct += (fx_server.argmax(1) == y).type(torch.float).sum().item()
    correct_train = correct / size
    loss, current = loss.item(), (batch_id + 1) * len(y)
    if batch_id == len(test_data) - 1:
        print("Using party B's all data", file=filename)
        print(f"Test-loss: {loss:>7f}  [{current:>5d}/{size:>5d}], Accuracy: {(100 * correct_train):>0.1f}%",
              file=filename)
        test_alone2_acc.append(100 * correct_train)
        test_alone2_loss.append(loss)
    return correct

def test_client(dataloader, t):
    client_model_1.eval()
    client_model_2.eval()
    correct = 0
    size = len(dataloader) * args.batch_size

    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X_1, X_2 = split_data(args.dataset, X)

        if args.dataset == 'utkface' or args.dataset == 'celeba':
            Y_1 = target[0].to(device)

        else:
            Y_1 = target.to(device)

        fx1 = client_model_1(X_1)
        fx2 = client_model_2(X_2)

        client1_fx = (fx1).clone().detach().requires_grad_(True)
        client2_fx = (fx2).clone().detach().requires_grad_(True)

        # Sending activations to server and receiving gradients from server
        correct = test_server(client1_fx, client2_fx, Y_1, batch_id, correct, size)

def test_partyB_seperate(dataloader, t):
    
    client_model_1.eval()
    client_model_2.eval()

    Decoder12.eval()

    correct_decoders = [0] * 7
    size = len(dataloader) * args.batch_size

    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X_1, X_2 = split_data(args.dataset, X)

        if args.dataset == 'utkface' or args.dataset == 'celeba':
            Y_1 = target[0].to(device)
        else:
            Y_1 = target.to(device)

        fx1 = client_model_1(X_1)

        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        partyB_decoders = []  

        for i, missing_ratio in enumerate(missing_ratios):

            if args.dataset == 'cifar10' or args.dataset == "imagenet" or args.dataset == 'utkface':
                D12 = Decoder12(fx1, X_2, (1 - missing_ratio), args.batch_size, 1)
            else:
                if missing_ratio == 0:
                    D12 = X_2
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

                        D12 = Decoder12(fx1, X_2, Missing_item_B, args.batch_size)
                        
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

                        D12 = Decoder12(fx1, X_2, Missing_item_B, args.batch_size)

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
                        
                        D12 = Decoder12(fx1, X_2, Missing_item_B, args.batch_size)
            
            fx12_d = client_model_2(D12)
            partyB_decoder = (fx12_d).clone().detach().requires_grad_(True)
            partyB_decoders.append(partyB_decoder)

        correct_decoders = test_partyB_server_seperate(partyB_decoders, Y_1, batch_id, correct_decoders, size)

def test_partyB_server_seperate(partyB_decoders, Y_1, batch_id, correct_decoders, size):

    server_model.eval()
    correct_decoders = correct_decoders

    loss = [0] * len(partyB_decoders) 
    
    for i, partyB_decoder in enumerate(partyB_decoders):
        optimizer_server.zero_grad()
        _, fx_server = server_model(partyB_decoder, partyB_decoder)
        loss[i] = criterion(fx_server, Y_1).item()
        correct_decoders[i] += (fx_server.argmax(1) == Y_1).type(torch.float).sum().item()
        current = (batch_id + 1) * len(Y_1)
    
    if batch_id == len(test_data) - 1:
        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        print("Party B completed by decoder",file=filename)
        for i, missing_ratio in enumerate(missing_ratios):
            correct_decoder_ratio = (correct_decoders[i] / size)
            print(f"Missing ratio: {missing_ratio}", file=filename)
            print(f"Test-loss: {loss[i]:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_decoder_ratio):>0.1f}%", file=filename)
            test_acc_seperate_partyB_decoders[i].append(100 * correct_decoder_ratio)
            test_loss_seperate_partyB_decoders[i].append(loss[i])
    
    return correct_decoders

def test_partyA_seperate(dataloader, t):

    client_model_1.eval()
    client_model_2.eval()
    Decoder21.eval()

    correct_decoders = [0] * 7

    size = len(dataloader) * args.batch_size

    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X_1, X_2 = split_data(args.dataset, X)

        if args.dataset == 'utkface' or args.dataset == 'celeba':
            Y_1 = target[0].to(device)
        else:
            Y_1 = target.to(device)

        fx2 = client_model_2(X_2)

        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        partyA_decoders = []  

        for i, missing_ratio in enumerate(missing_ratios):

            if args.dataset == 'cifar10' or args.dataset == "imagenet" or args.dataset == 'utkface':
                D21 = Decoder21(fx2, X_1, (1 - missing_ratio), args.batch_size, 0)
            else:
                if missing_ratio == 0:
                    D21 = X_1
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

                        D21 = Decoder21(fx2, X_1, Missing_item_A, args.batch_size)
                        
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
                        D21 = Decoder21(fx2, X_1, Missing_item_A, args.batch_size)
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

                        D21 = Decoder21(fx2, X_1, Missing_item_A, args.batch_size)
            fx21_d = client_model_1(D21)
            partyA_decoder = (fx21_d).clone().detach().requires_grad_(True)
            partyA_decoders.append(partyA_decoder)
        
        correct_decoders = test_partyA_server_seperate(partyA_decoders, Y_1, batch_id, correct_decoders, size)
        
def test_partyA_server_seperate(partyA_decoders, Y_1, batch_id, correct_decoders, size):

    server_model.eval()
    correct_decoders = correct_decoders

    loss = [0] * len(partyA_decoders) 
    for i, partyA_decoder in enumerate(partyA_decoders):
        optimizer_server.zero_grad()
        _, fx_server = server_model(partyA_decoder, partyA_decoder)
        loss[i] = criterion(fx_server, Y_1).item()
        correct_decoders[i] += (fx_server.argmax(1) == Y_1).type(torch.float).sum().item()
        current = (batch_id + 1) * len(Y_1)
    if batch_id == len(test_data) - 1:
        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        print("Party A completed by decoder",file=filename)
        for i, missing_ratio in enumerate(missing_ratios):
            correct_decoder_ratio = (correct_decoders[i] / size)
            print(f"Missing ratio: {missing_ratio}", file=filename)
            print(f"Test-loss: {loss[i]:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_decoder_ratio):>0.1f}%", file=filename)
            test_acc_seperate_partyA_decoders[i].append(100 * correct_decoder_ratio)
            test_loss_seperate_partyA_decoders[i].append(loss[i])
    return correct_decoders

def test_alone1_client(dataloader, t):
    client_model_1.eval()
    client_model_2.eval()

    correct = 0
    size = len(dataloader) * args.batch_size

    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X_1, X_2 = split_data(args.dataset, X)

        if args.dataset == 'utkface' or args.dataset == 'celeba':
            Y_1 = target[0].to(device)
        else:
            Y_1 = target.to(device)

        fx1 = client_model_1(X_1)
        client1_fx = (fx1).clone().detach().requires_grad_(True)

        correct = test_alone1_server(client1_fx, Y_1, batch_id, correct, size, t)

def test_alone2_client(dataloader, t):
    client_model_1.eval()
    client_model_2.eval()
    correct = 0
    size = len(dataloader) * args.batch_size

    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X_1, X_2 = split_data(args.dataset, X)

        if args.dataset == 'utkface' or args.dataset == 'celeba':
            Y_1 = target[0].to(device)
        else:
            Y_1 = target.to(device)

        fx2 = client_model_2(X_2)
        client2_fx = (fx2).clone().detach().requires_grad_(True)

        correct = test_alone2_server(client2_fx, Y_1, batch_id, correct, size, t)

def test_two_partys_decoder(dataloader, t):

    client_model_1.eval()
    client_model_2.eval()
    Decoder12.eval()
    Decoder21.eval()

    correct_partyA_completedByDecoders = [0] * 7
    correct_partyB_completedByDecoders = [0] * 7
    correct_vote_basedOnPartyA = [0] * 7 
    correct_vote_basedOnPartyB = [0] * 7
    
    size = len(dataloader) * args.batch_size

    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X_1, X_2 = split_data(args.dataset, X)

        if args.dataset == 'utkface' or args.dataset == 'celeba':
            Y_1 = target[0].to(device)
        else:
            Y_1 = target.to(device)

        fx2 = client_model_2(X_2)
        client2_fx = (fx2).clone().detach().requires_grad_(True)

        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        partyA_decoders = []  

        for i, missing_ratioA in enumerate(missing_ratios):

            if args.dataset == 'cifar10' or args.dataset == "imagenet" or args.dataset == 'utkface':
                D21 = Decoder21(fx2, X_1, (1 - missing_ratioA), args.batch_size, 0)
            else:
                if missing_ratioA == 0:
                    D21 = X_1
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
                        Missing_item_A = missing_ratio_to_item_A.get(missing_ratioA, None)  

                        D21 = Decoder21(fx2, X_1, Missing_item_A, args.batch_size)
                        
                    elif args.dataset == 'bank':
                        missing_ratio_to_item_A = {
                            0.1: 1,
                            0.3: 2,
                            0.5: 3,
                            0.7: 4,
                            0.9: 5,
                            1.0: 29
                        }
                        Missing_item_A = missing_ratio_to_item_A.get(missing_ratioA, None)  
                        D21 = Decoder21(fx2, X_1, Missing_item_A, args.batch_size)
                    elif args.dataset == 'avazu':
                        missing_ratio_to_item_A = {
                            0.1: 2,
                            0.3: 3,
                            0.5: 4,
                            0.7: 14,
                            0.9: 24,
                            1.0: 39
                        }
                        Missing_item_A = missing_ratio_to_item_A.get(missing_ratioA, None)  

                        D21 = Decoder21(fx2, X_1, Missing_item_A, args.batch_size)

            fx21_d = client_model_1(D21)
            partyA_decoder = (fx21_d).clone().detach().requires_grad_(True)
            partyA_decoders.append(partyA_decoder)

        fx1 = client_model_1(X_1)
        client1_fx = (fx1).clone().detach().requires_grad_(True)

        partyB_decoders = []  

        for i, missing_ratioB in enumerate(missing_ratios):

            if args.dataset == 'cifar10' or args.dataset == "imagenet" or args.dataset == 'utkface':
                D12 = Decoder12(fx1, X_2, (1 - missing_ratioB), args.batch_size, 1)
            else:
                if missing_ratioB == 0:
                    D12 = X_2
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
                        Missing_item_B = missing_ratio_to_item_B.get(missing_ratioB, None)  

                        D12 = Decoder12(fx1, X_2, Missing_item_B, args.batch_size)
                        
                    elif args.dataset == 'bank':

                        missing_ratio_to_item_B = {
                            0.1: 1,
                            0.3: 2,
                            0.5: 3,
                            0.7: 4,
                            0.9: 5,
                            1.0: 28
                        }
                        Missing_item_B = missing_ratio_to_item_B.get(missing_ratioB, None) 

                        D12 = Decoder12(fx1, X_2, Missing_item_B, args.batch_size)

                    elif args.dataset == 'avazu':
                
                        missing_ratio_to_item_B = {
                            0.1: 1,
                            0.3: 2,
                            0.5: 3,
                            0.7: 13,
                            0.9: 23,
                            1.0: 38
                        }
                        Missing_item_B = missing_ratio_to_item_B.get(missing_ratioB, None)  
                        
                        D12 = Decoder12(fx1, X_2, Missing_item_B, args.batch_size)

            fx12_d = client_model_2(D12)
            partyB_decoder = (fx12_d).clone().detach().requires_grad_(True)
            partyB_decoders.append(partyB_decoder)


        correct_partyA_completedByDecoders,\
              correct_partyB_completedByDecoders,\
                  correct_vote_basedOnPartyA, correct_vote_basedOnPartyB = test_server_two_parties_decoders(client1_fx, client2_fx, partyA_decoders, partyB_decoders,
                                                    correct_partyA_completedByDecoders,correct_partyB_completedByDecoders,
                                                    correct_vote_basedOnPartyA, correct_vote_basedOnPartyB,
                                                    Y_1, batch_id, size)

def test_server_two_parties_decoders(client1_fx, client2_fx, partyA_decoders, partyB_decoders,
                                                    correct_partyA_completedByDecoders,correct_partyB_completedByDecoders,
                                                    correct_vote_basedOnPartyA, correct_vote_basedOnPartyB,
                                                    Y_1, batch_id, size):

    server_model.eval()

    correct_partyA_completedByDecoders = correct_partyA_completedByDecoders
    correct_partyB_completedByDecoders = correct_partyB_completedByDecoders
    correct_vote_basedOnPartyA = correct_vote_basedOnPartyA
    correct_vote_basedOnPartyB = correct_vote_basedOnPartyB

    loss_decoder_basedOnPartyA = [0] * len(partyB_decoders) 
    for i, partyB_decoder in enumerate(partyB_decoders):

        optimizer_server.zero_grad()
        
        _, fx_server_decoder_basedOnPartyA = server_model(client1_fx, partyB_decoder)
        loss_decoder_basedOnPartyA[i] = criterion(fx_server_decoder_basedOnPartyA, Y_1).item()
        correct_partyB_completedByDecoders[i] += (fx_server_decoder_basedOnPartyA.argmax(1) == Y_1).type(torch.float).sum().item()
        current = (batch_id + 1) * len(Y_1)
        
        _, fx_server_wholeA = server_model(client1_fx, client1_fx)
        pred_wholeA = fx_server_wholeA.argmax(1)
        
        _, fx_server_BbyDecoder = server_model(partyB_decoder, partyB_decoder)
        pred_BbyDecoder = fx_server_BbyDecoder.argmax(1)
        
        pred_avg = fx_server_decoder_basedOnPartyA.argmax(1)
        
        predictions = torch.stack([pred_avg, pred_wholeA, pred_BbyDecoder], dim=0)  
        final_predictions, _ = torch.mode(predictions, dim=0) 
        
        correct_vote_basedOnPartyA[i] += (final_predictions == Y_1).type(torch.float).sum().item()

    loss_decoder_basedOnPartyB = [0] * len(partyA_decoders) 
    for i, partyA_decoder in enumerate(partyA_decoders):
        
        optimizer_server.zero_grad()

        _, fx_server_decoder_basedOnPartyB = server_model(client2_fx, partyA_decoder)
        loss_decoder_basedOnPartyB[i] = criterion(fx_server_decoder_basedOnPartyB, Y_1).item()
        correct_partyA_completedByDecoders[i] += (fx_server_decoder_basedOnPartyB.argmax(1) == Y_1).type(torch.float).sum().item()
        current = (batch_id + 1) * len(Y_1)
        
        _, fx_server_wholeB = server_model(client2_fx, client2_fx)
        pred_wholeB = fx_server_wholeB.argmax(1)
        
        _, fx_server_AbyDecoder = server_model(partyA_decoder, partyA_decoder)
        pred_AbyDecoder = fx_server_AbyDecoder.argmax(1)
        
        pred_avg = fx_server_decoder_basedOnPartyB.argmax(1)
        
        predictions = torch.stack([pred_avg, pred_wholeB, pred_AbyDecoder], dim=0)  
        final_predictions, _ = torch.mode(predictions, dim=0) 
        
        correct_vote_basedOnPartyB[i] += (final_predictions == Y_1).type(torch.float).sum().item()
        
    if batch_id == len(test_data) - 1:
        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        for i, missing_ratio in enumerate(missing_ratios):
            print(f"Missing ratio: {missing_ratio}", file=filename)
            correct_partyB_completedByDecoder_ratio = correct_partyB_completedByDecoders[i] / size
            correct_partyA_completedByDecoder_ratio = correct_partyA_completedByDecoders[i] / size

            correct_both_decoder_ratio = (correct_partyB_completedByDecoder_ratio + correct_partyA_completedByDecoder_ratio)/2
            correct_vote_ratio = (correct_vote_basedOnPartyA[i] + correct_vote_basedOnPartyB[i])/(2*size)
            
            print((
                f"----For partyA is completed and partyB is completed by decoder----\n"
                f"    Test-Accuracy: {(100 * correct_partyB_completedByDecoder_ratio):>0.1f}%\n"
                f"    Test-Loss: {loss_decoder_basedOnPartyA[i]:>7f}\n"
                f"----For partyB is completed and partyA is completed by decoder----\n"
                f"    Test-Accuracy: {(100 * correct_partyA_completedByDecoder_ratio):>0.1f}%\n"
                f"    Test-Loss: {loss_decoder_basedOnPartyB[i]:>7f}\n"
                f"----For the average accuracy of the two situations----\n"
                f"    Test-Accuracy: {(100 * correct_both_decoder_ratio):>0.1f}%\n"
                f"----For the average accuracy of the two situations (vote)----\n"
                f"    Test-Accuracy: {(100 * correct_vote_ratio):>0.1f}%\n"
                f"Processing Progress: [{current:>5d}/{size:>5d}]\n"
            ), file=filename)

            test_acc_all_completedByDecoders[i].append(100 * correct_both_decoder_ratio)
            
    return correct_partyA_completedByDecoders, correct_partyB_completedByDecoders,\
            correct_vote_basedOnPartyA, correct_vote_basedOnPartyB

def find_top_k_average(test_data, k=5):

    if len(test_data) < k:
        return None

    top_k_values = sorted(test_data, reverse=True)[:k]
    return np.mean(top_k_values)

if __name__ == '__main__':
    
    root_path = '.'
    data_path = os.path.join(root_path, 'Dataset').replace('\\', '/')
    save_path = f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = open(f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/A_training_process.txt', 'w+')

    ### Load data 
    if args.dataset == 'utkface' or args.dataset == 'celeba':
                
        train_data_nondataloader, test_data, num_classes, num_classes2, channel, hidden = load_data(args.dataset, args.attributes, data_path, args.batch_size)
        
    elif args.dataset =='MIMIC' or args.dataset == 'avazu' or args.dataset == 'bank':
        train_data_nondataloader, test_data, input_dim1, input_dim2, num_classes = load_data_tabular(args.dataset, args.batch_size)

    elif args.dataset == "imagenet":
        train_data_nondataloader, test_data, num_classes, channel, hidden = get_tinyimagenet_bothloader(batch_size=args.batch_size, shuffle=True, seed=args.seed, device=device, classes = args.classes)
    else:
        train_data_nondataloader, test_data, num_classes, channel, hidden = gen_dataset(args.dataset, data_path, device = device)

    # Define model
    if args.dataset =='MIMIC' or args.dataset == 'avazu' or args.dataset == 'bank':
        client_model_1, client_model_2, server_model = def_tabular_model(dataset= args.dataset, level= args.level, input_dim1=input_dim1, input_dim2=input_dim2, num_classes=num_classes, num_cutlayer=args.num_cutlayer, mode='sum')
        _, _, server_test_model = def_tabular_model(dataset= args.dataset, level= args.level, input_dim1=input_dim1, input_dim2=input_dim2, num_classes=num_classes, num_cutlayer=args.num_cutlayer, mode='standalone')

    else:

        ImageModel1= ImageModel(dataset= args.dataset, hidden=hidden, num_cutlayer=args.num_cutlayer, num_classes=num_classes, mode='sum', device=device)
        client_model_1, client_model_2, server_model = ImageModel1.GetModel()

        ImageModel2= ImageModel(dataset= args.dataset, hidden=hidden, num_cutlayer=args.num_cutlayer, num_classes=num_classes, mode='standalone', device=device)
        _, _, server_test_model = ImageModel2.GetModel()

    # Define decoder/XCom
    if args.dataset == 'mnist' or args.dataset == 'fmnist':  
        
        Decoder12 = Generator_mnist(channel=channel, shape_img=28, batchsize=args.batch_size, g_in=args.num_cutlayer, iters=1).to(device)     
        Decoder21 = Generator_mnist(channel=channel, shape_img=28, batchsize=args.batch_size, g_in=args.num_cutlayer, iters=0).to(device)

    elif args.dataset =='MIMIC' or args.dataset == 'avazu' or args.dataset == 'bank':
        if args.level == 1:
            Decoder21 = Generator_tabular_1(in_dim=input_dim1, n_hidden_1=256,  n_hidden_2=args.num_cutlayer).to(device)
            Decoder12 = Generator_tabular_1(in_dim=input_dim2, n_hidden_1=256,  n_hidden_2=args.num_cutlayer).to(device)
        elif args.level == 2:
            Decoder21 = Generator_tabular_2(in_dim=input_dim1, n_hidden_1=64,  n_hidden_2=512, n_hidden_3=args.num_cutlayer).to(device)
            Decoder12 = Generator_tabular_2(in_dim=input_dim2, n_hidden_1=64,  n_hidden_2=512, n_hidden_3=args.num_cutlayer).to(device)
        elif args.level == 3:
            Decoder21 = Generator_tabular_3(in_dim=input_dim1, n_hidden_1=64, n_hidden_2=128, n_hidden_3=256,n_hidden_4=args.num_cutlayer).to(device)
            Decoder12 = Generator_tabular_3(in_dim=input_dim2, n_hidden_1=64, n_hidden_2=128, n_hidden_3=256,n_hidden_4=args.num_cutlayer).to(device)
        elif args.level == 4:
            Decoder21 = Generator_tabular_4(in_dim=input_dim1, n_hidden_1=32,  n_hidden_2=128, n_hidden_3=512, n_hidden_4=1024, n_hidden_5=args.num_cutlayer).to(device)
            Decoder12 = Generator_tabular_4(in_dim=input_dim2, n_hidden_1=32,  n_hidden_2=128, n_hidden_3=512, n_hidden_4=1024, n_hidden_5=args.num_cutlayer).to(device)
    
    else:   
        Decoder12 = Generator_cifar(channel=channel, shape_img=32, batchsize=args.batch_size, g_in=args.num_cutlayer, iters=1).to(device)
        Decoder21 = Generator_cifar(channel=channel, shape_img=32, batchsize=args.batch_size, g_in=args.num_cutlayer, iters=0).to(device)

    # criterion & optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer_client1 = torch.optim.Adam(client_model_1.parameters(), lr=args.lr, foreach=False )
    optimizer_client2 = torch.optim.Adam(client_model_2.parameters(), lr=args.lr, foreach=False )
    optimizer_server  = torch.optim.Adam(server_model.parameters(),   lr=args.lr, foreach=False )  

    optimizer_decoder12 = torch.optim.Adam(Decoder12.parameters(), lr=args.lr, foreach=False )
    optimizer_decoder21 = torch.optim.Adam(Decoder21.parameters(), lr=args.lr, foreach=False )

    # record results
    train_overlap_loss = []
    train_overlap_acc = []
    train_complete1_loss = []
    train_complete1_acc = []
    train_complete2_loss = []
    train_complete2_acc = []
    
    test_acc = []
    test_alone1_acc = []
    test_alone2_acc = []
    
    test_loss = []
    test_alone1_loss = []
    test_alone2_loss = []

    test_acc_seperate_partyA_decoders = [[] for _ in range(7)]   
    test_loss_seperate_partyA_decoders = [[] for _ in range(7)]  
    test_acc_seperate_partyB_decoders = [[] for _ in range(7)]   
    test_loss_seperate_partyB_decoders = [[] for _ in range(7)]   
    test_acc_all_completedByDecoders = [[] for _ in range(7)] 

    # start training
    for t in range(args.epochs):

        if args.dataset == 'cifar10' or args.dataset == "imagenet" or args.dataset == 'utkface' or args.dataset =='MIMIC' or args.dataset == 'bank' or args.dataset == 'avazu':
            train_data = get_train(train_data_nondataloader, args.batch_size, args.seed, device) 
        
        print(f"Epoch {t + 1}\n-------------------------------", file=filename)
        
        train_client(train_data, client_model_1, client_model_2, t)
    print("Done!", file=filename)

    # save decoders
    save_path_D12 = f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/Decoder12.pth'
    torch.save(Decoder12, save_path_D12)
    save_path_D21 = f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/Decoder21.pth'
    torch.save(Decoder21, save_path_D21)

    print("----Final result----", file=filename)
    print("----Find the Max result----", file=filename)

    print("Max--test_acc: ", np.array(test_acc).max(), file=filename)
    print("Max--test_alone1_acc: ", np.array(test_alone1_acc).max(), file=filename)
    print("Max--test_alone2_acc: ", np.array(test_alone2_acc).max(), file=filename)
    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        print(f"Missing_ratio: {missing_ratio}", file=filename)
        print("Max--test_acc_seperate_partyB_decoder: ", np.array(test_acc_seperate_partyB_decoders[i]).max(), file=filename)
        print("Max--test_acc_seperate_partyA_decoder: ", np.array(test_acc_seperate_partyA_decoders[i]).max(), file=filename)
        print(f"Max--test_acc_all_completedByDecoder: {np.array(test_acc_all_completedByDecoders[i]).max()}, \n",file=filename)

    print("----Final result----", file=filename)
    print("----Find the top k result----", file=filename)
    print("topk--test_acc: ", find_top_k_average(test_acc), file=filename)
    print("topk--test_alone1_acc: ", find_top_k_average(test_alone1_acc), file=filename)
    print("topk--test_alone2_acc: ", find_top_k_average(test_alone2_acc), file=filename)
    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        print(f"Missing_ratio: {missing_ratio}", file=filename)
        print("topk--test_acc_seperate_partyB_decoder: ", find_top_k_average(test_acc_seperate_partyB_decoders[i]), file=filename)
        print("topk--test_acc_seperate_partyA_decoder: ", find_top_k_average(test_acc_seperate_partyA_decoders[i]), file=filename)

        average_decoder = find_top_k_average(test_acc_all_completedByDecoders[i])
        print(f"topk--test_acc_all_completedByDecoder: {average_decoder},", file=filename)

    #plt
    x = np.arange(len(train_overlap_acc))
    plt.plot(x, train_overlap_acc, label='train_overlap_acc')
    plt.plot(x, train_complete1_acc, label='train_complete1_acc')
    plt.plot(x, train_complete2_acc, label='train_complete2_acc')

    plt.xlabel('epoch',   fontsize=19)
    plt.ylabel('ACC',   fontsize=19)
    plt.title(f'{args.dataset}_sample{args.num_overlap}',   fontsize=20)
    plt.legend()
    plt.savefig(f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/A_train_ACC.png')
    plt.close()

    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):

        x = np.arange(len(train_overlap_acc))
        plt.plot(x, test_acc,  label='test_overlap_acc' ,linestyle='-', marker='o', linewidth=0.8, markersize=2)
        plt.plot(x, test_alone1_acc, label='test_alone1_acc',linestyle='-', marker='o', linewidth=0.8, markersize=2)
        plt.plot(x, test_alone2_acc, label='test_alone2_acc',linestyle='-', marker='o', linewidth=0.8, markersize=2)

        plt.plot(x, test_acc_seperate_partyB_decoders[i],  label='test_acc_seperate_partyB_decoder', linestyle='--', marker='s', linewidth=0.8, markersize=2)
        plt.plot(x, test_acc_seperate_partyA_decoders[i], label='test_acc_seperate_partyA_decoder', linestyle='--', marker='s', linewidth=0.8, markersize=2)
        plt.plot(x, test_acc_all_completedByDecoders[i],  label='test_acc_all_completedByDecoder', linestyle='-.', marker='*', linewidth=0.8, markersize=2) 

        plt.xlabel('epoch', fontsize=19)
        plt.ylabel('ACC', fontsize=19)
        plt.title(f'{args.dataset}_sample{args.num_overlap}_Missing{missing_ratio}',   fontsize=20)
        plt.legend()
        plt.savefig(f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/A_test_acc_Missing{missing_ratio}.png')
        plt.close()
    
    x = np.arange(len(train_overlap_loss))
    plt.plot(x, train_overlap_loss, label='train_overlap_loss')
    plt.plot(x, train_complete1_loss, label='train_complete1_loss')
    plt.plot(x, train_complete2_loss, label='train_complete2_loss')

    plt.xlabel('epoch',   fontsize=19)
    plt.ylabel('Loss',   fontsize=19)
    plt.title(f'{args.dataset}_sample{args.num_overlap}',   fontsize=20)
    plt.legend()
    plt.savefig(f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/A_train_Loss.png')
    plt.close()
    
    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):

        x = np.arange(len(train_overlap_loss))
        plt.plot(x, test_loss,  label='test_overlap_loss',linestyle='-', marker='o', linewidth=0.8, markersize=2)
        plt.plot(x, test_alone1_loss, label='test_alone1_loss',linestyle='-', marker='o', linewidth=0.8, markersize=2)
        plt.plot(x, test_alone2_loss, label='test_alone2_loss',linestyle='-', marker='o', linewidth=0.8, markersize=2)
        plt.plot(x, test_loss_seperate_partyB_decoders[i],  label='test_loss_seperate_partyB_decoder', linestyle='-.', marker='*', linewidth=0.8, markersize=2) 
        plt.plot(x, test_loss_seperate_partyA_decoders[i], label='test_loss_seperate_partyA_decoder', linestyle='-.', marker='*', linewidth=0.8, markersize=2) 

        plt.xlabel('epoch',   fontsize=19)
        plt.ylabel('Loss',   fontsize=19)
        plt.title(f'{args.dataset}_sample{args.num_overlap}_Missing{missing_ratio}',   fontsize=20)
        plt.legend()
        plt.savefig(f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/A_test_loss_Missing{missing_ratio}.png')
        plt.close()

    plt.xlabel('epoch',   fontsize=19)
    plt.ylabel('Loss',   fontsize=19)
    plt.title(f'{args.dataset}_sample{args.num_overlap}',   fontsize=20)
    plt.legend()
    plt.savefig(f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/Lambda&Center_Loss.png')
    plt.close()
    
    np.save(f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/test_acc.npy', test_acc) 
    np.save(f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/test_alone1_acc.npy', test_alone1_acc) 
    np.save(f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/test_alone2_acc.npy', test_alone2_acc) 

    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        np.save(f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/test_acc_seperate_partyB_decoder_Missing{missing_ratio}.npy', test_acc_seperate_partyB_decoders[i]) 
        np.save(f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/test_acc_seperate_partyA_decoder_Missing{missing_ratio}.npy', test_acc_seperate_partyA_decoders[i]) 
        np.save(f'Result/Results_X-VFL/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.lambda1}-l2{args.lambda2}/test_acc_all_completedByDecoder_Missing{missing_ratio}.npy', test_acc_all_completedByDecoders[i]) 

