import os
import copy
from torch import nn
import numpy as np
from models import *
from utils import *
from parse import args
import matplotlib.pyplot as plt
from Decoder import *
import fedcvt_repr_estimator as estimator
import torch
import torch.nn.functional as F

# Define a random_seed
set_random_seed(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
device = torch.device(f"cuda") 

def transpose(x):
    return x if x is None else torch.transpose(x, 0, 1)

# Train_Client Side Program
def train_client(dataloader, client_model_1U, client_model_1C,
                client_model_2U, client_model_2C,
                server_model_partyA, server_model_partyB,
                server_model_partyAB, t):

    client_model_1U.train()
    client_model_1C.train()
    client_model_2U.train()
    client_model_2C.train()
    server_model_partyA.train()
    server_model_partyB.train()
    server_model_partyAB.train()

    correct_A = 0
    correct_B = 0
    correct_Fed = 0

    size = len(dataloader)*args.batch_size

    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X_1, X_2 = split_data(args.dataset, X)
        
        if args.dataset == 'utkface' or args.dataset == 'celeba':
            Y_1 = target[0].to(device)
            Y_2 = target[1].view(-1, 1).to(device)
        else:
            Y_1 = target.to(device)
        
        X_1_overlap   = X_1[:args.num_overlap]
        X_2_overlap   = X_2[:args.num_overlap]
        X_1_complete1 = X_1[args.num_overlap:int((args.batch_size - args.num_overlap)*args.nonoverlap1_ratio + args.num_overlap)]
        X_2_complete1 = X_2[args.num_overlap:int((args.batch_size - args.num_overlap)*args.nonoverlap1_ratio + args.num_overlap)]
        
        X_1_complete2 = X_1[int((args.batch_size - args.num_overlap)*(args.nonoverlap1_ratio) + args.num_overlap):args.batch_size]
        X_2_complete2 = X_2[int((args.batch_size - args.num_overlap)*(args.nonoverlap1_ratio) + args.num_overlap):args.batch_size]
        
        Y_overlap   = Y_1[:args.num_overlap]
        Y_complete1 = Y_1[args.num_overlap:int((args.batch_size - args.num_overlap)*args.nonoverlap1_ratio + args.num_overlap)]
        Y_complete2 = Y_1[int((args.batch_size - args.num_overlap)*(args.nonoverlap1_ratio) + args.num_overlap):args.batch_size]

        # step 1: learn representation

        # step 1.1 slove the situation of overlap
        fx1_overlap_U = client_model_1U(X_1_overlap)
        fx1_overlap_C = client_model_1C(X_1_overlap)
        fx2_overlap_U = client_model_2U(X_2_overlap)
        fx2_overlap_C = client_model_2C(X_2_overlap)

        # C goes first and U goes after
        
        fx1_overlap_U_C = torch.cat([fx1_overlap_U, fx1_overlap_C], dim=1)
        fx2_overlap_U_C = torch.cat([fx2_overlap_U, fx2_overlap_C], dim=1)
        
        # step 1.2 slove the situation of complete1 (no partyA)
        fx2_complete1_U = client_model_2U(X_2_complete1)
        fx2_complete1_C = client_model_2C(X_2_complete1)
        fx2_complete1_U_C = torch.cat([fx2_complete1_U, fx2_complete1_C], dim=1)
        # C goes first and U goes after
        
        # step 1.3 slove the situation of complete2 (no partyB)
        fx1_complete2_U = client_model_1U(X_1_complete2)
        fx1_complete2_C = client_model_1C(X_1_complete2)
        fx1_complete2_U_C = torch.cat((fx1_complete2_U, fx1_complete2_C), dim=1)
        # C goes first and U goes after

        R_C_A = torch.cat((fx1_overlap_C, fx1_complete2_C), dim=0)

        # use B to complete A
        W_hg = None
        fx1_complete1_U_C = estimator.AttentionBasedRepresentationEstimator.estimate_host_reprs_for_guest_party(
            Ug_comm = fx2_complete1_C, Ug_uniq = fx2_complete1_U, Ug_overlap_uniq = fx2_overlap_U,
            Uh_overlap_uniq = fx1_overlap_U, Uh_all_comm = R_C_A, sharpen_temperature=args.tem,
            W_gh=transpose(W_hg), using_uniq=True, using_comm=True)

        split_idx = fx1_complete1_U_C.shape[1] // 2

        fx1_complete1_U = fx1_complete1_U_C[:, :split_idx]  
        fx1_complete1_C = fx1_complete1_U_C[:, split_idx:]  
        
        R_C_B  = torch.cat((fx2_overlap_C, fx2_complete1_C),dim=0)
        # use A to complete B
        fx2_complete2_U_C = estimator.AttentionBasedRepresentationEstimator.estimate_guest_reprs_for_host_party(
            Uh_comm = fx1_complete2_C, Uh_uniq = fx1_complete2_U, Uh_overlap_uniq = fx1_overlap_U,
            Ug_overlap_uniq = fx2_overlap_U, Ug_all_comm = R_C_B, sharpen_tempature=args.tem,
            W_hg=None, using_uniq=True, using_comm=True)
        
        split_idx = fx2_complete2_U_C.shape[1] // 2

        fx2_complete2_U = fx2_complete2_U_C[:, :split_idx]  
        fx2_complete2_C = fx2_complete2_U_C[:, split_idx:]  

        # step 3: estimate pseudo labels (ignored here)

        # step 4: Feed χA, χB and χ to classifiers f A, f B and f AB respectively for cross-view training

        fx1_U_C = torch.cat((fx1_overlap_U_C, fx1_complete1_U_C, fx1_complete2_U_C), dim=0)
        fx2_U_C = torch.cat((fx2_overlap_U_C, fx2_complete1_U_C, fx2_complete2_U_C), dim=0)
        fx1_C = torch.cat((fx1_overlap_C, fx1_complete1_C, fx1_complete2_C), dim=0)
        fx1_U = torch.cat((fx1_overlap_U, fx1_complete1_U, fx1_complete2_U), dim=0)
        fx2_C = torch.cat((fx2_overlap_C, fx2_complete1_C, fx2_complete2_C), dim=0)
        fx2_U = torch.cat((fx2_overlap_U, fx2_complete1_U, fx2_complete2_U), dim=0)
        _, server_fx1 = server_model_partyA(fx1_U, fx1_C)
        _, server_fx2 = server_model_partyB(fx2_U, fx2_C)
        _, server_fx12 = server_model_partyAB(fx2_U_C, fx1_U_C)
       
        # step 5: Compute loss: Lob j = Lf ed + LA + LB + λ1LAB  diff + λ2LA  diff + λ3LB  diff + λ4LA  sim + λ5LB  sim;
        L_diff_AB = F.mse_loss(fx1_overlap_C, fx2_overlap_C, reduction="mean")

        R_U_A = torch.cat((fx1_overlap_U, fx1_complete2_U), dim=0)
        R_U_B = torch.cat((fx2_overlap_U, fx2_complete1_U),dim=0)

        num_samples = torch.tensor(R_U_A.shape[0]).float()
        L_sim_A = torch.norm(torch.matmul(R_U_A, torch.transpose(R_C_A, 0, 1))) / num_samples
        
        num_samples = torch.tensor(R_U_B.shape[0]).float()
        L_sim_B = torch.norm(torch.matmul(R_U_B, torch.transpose(R_C_B, 0, 1))) / num_samples
        
        fx1_overlap_U_C_estimate = estimator.AttentionBasedRepresentationEstimator.estimate_host_reprs_for_guest_party(
            Ug_comm = fx2_overlap_C, Ug_uniq = fx2_overlap_U, Ug_overlap_uniq = fx2_overlap_U,
            Uh_overlap_uniq = fx1_overlap_U, Uh_all_comm = R_C_A, sharpen_temperature=args.tem,
            W_gh=transpose(W_hg), using_uniq=True, using_comm=True)

        fx2_overlap_U_C_estimate = estimator.AttentionBasedRepresentationEstimator.estimate_guest_reprs_for_host_party(
            Uh_comm = fx1_overlap_C, Uh_uniq = fx1_overlap_U, Uh_overlap_uniq = fx1_overlap_U,
            Ug_overlap_uniq = fx2_overlap_U, Ug_all_comm = R_C_B, sharpen_tempature=args.tem,
            W_hg=None, using_uniq=True, using_comm=True)

        L_diff_A = F.mse_loss(fx1_overlap_U_C_estimate, fx1_overlap_U_C, reduction="mean")
        L_diff_B = F.mse_loss(fx2_overlap_U_C_estimate, fx2_overlap_U_C, reduction="mean")

        L_ce_A = criterion(server_fx1, Y_1) 
        L_ce_B = criterion(server_fx2, Y_1) 
        L_ce_fed = criterion(server_fx12, Y_1) 

        Loss = L_ce_fed + L_ce_A + L_ce_B + args.CVTL1*L_diff_AB + args.CVTL2*L_diff_A + \
                args.CVTL3*L_diff_B + args.CVTL4*L_sim_A + args.CVTL5*L_sim_B

        # step 6: update the model
        optimizer_client1U.zero_grad()
        optimizer_client1U.zero_grad()
        optimizer_client2U.zero_grad()
        optimizer_client2C.zero_grad()
        optimizer_server_partyA.zero_grad()
        optimizer_server_partyB.zero_grad()
        optimizer_server_partyAB.zero_grad()

        Loss.backward()

        optimizer_client1U.step()
        optimizer_client1U.step()
        optimizer_client2U.step()
        optimizer_client2C.step()
        optimizer_server_partyA.step()
        optimizer_server_partyB.step()
        optimizer_server_partyAB.step()

        # step 7: record the correct number and the para of the models
        # save the models to the exactly different path
        correct_A += (server_fx1.argmax(1) == Y_1).type(torch.float).sum().item()
        correct_B += (server_fx2.argmax(1) == Y_1).type(torch.float).sum().item()
        correct_Fed += (server_fx12.argmax(1) == Y_1).type(torch.float).sum().item()

        if batch_id == args.num_batch-1:

            #acc
            correct_A_ratio = correct_A / size
            correct_B_ratio = correct_B / size
            correct_Fed_ratio = correct_Fed / size

            current = (batch_id + 1) *args.batch_size
            Loss = Loss.item()
            print("---training precessing---")
            print(f"---epoch: {t+1}---", file=filename)
            print(
                f"Head A Accuracy:  {(100 * correct_A_ratio):>0.1f}%,\n"
                f"Head B Accuracy:  {(100 * correct_B_ratio):>0.1f}%,\n"
                f"Head Fed Accuracy: {(100 * correct_Fed_ratio):>0.1f}%,\n",
                file=filename,
            )
            print(f"Loss: {Loss:>7f},\n"
                  f"Processing: [{current:>5d}/{size:>5d}]",file=filename
                 )
            
            train_A_acc.append(100 * correct_A_ratio)
            train_B_acc.append(100 * correct_B_ratio)
            train_Fed_acc.append(100 * correct_Fed_ratio)

            train_loss.append(Loss)

            #test
            test_client(test_data, t)
       
            test_two_partys_zero(test_data, t)
        
        if (t+1) %10 ==0 or t == args.epochs-1:
            if batch_id == args.num_batch-1:  

                save_path1_U = f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/client1_U_epoch{t+1}.pth'
                save_path1_C = f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/client1_C_epoch{t+1}.pth'
                
                save_path2_U = f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/client2_U_epoch{t+1}.pth'
                save_path2_C = f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/client2_C_epoch{t+1}.pth'
                
                torch.save(client_model_1U, save_path1_U)
                torch.save(client_model_1C, save_path1_C)
                torch.save(client_model_2U, save_path2_U)
                torch.save(client_model_2C, save_path2_C)

                save_path_server_model_partyA = f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/server_model_partyA_epoch{t+1}.pth'
                torch.save(server_model_partyA, save_path_server_model_partyA)
                save_path_server_model_partyB = f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/server_model_partyB_epoch{t+1}.pth'
                torch.save(server_model_partyB, save_path_server_model_partyB)
                save_path_server_model_partyAB = f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/server_model_partyAB_epoch{t+1}.pth'
                torch.save(server_model_partyAB, save_path_server_model_partyAB)
        
        # break        
        if batch_id >= args.num_batch-1:
            break
            
# Test_Server Side Program
def test_server(client1_fx_U, client1_fx_C, 
                client2_fx_U, client2_fx_C,
                Y_1, batch_id, test_correct_vote, test_correct_fed, size,
                test_correct_partyA, test_correct_partyB):
    server_model_partyA.eval()
    server_model_partyB.eval()
    server_model_partyAB.eval()


    test_correct_vote = test_correct_vote
    test_correct_fed = test_correct_fed
    test_correct_partyA = test_correct_partyA
    test_correct_partyB = test_correct_partyB

    # train and update
    optimizer_server_partyA.zero_grad()
    optimizer_server_partyB.zero_grad() 
    optimizer_server_partyAB.zero_grad()

    _, fx_server_partyA = server_model_partyA(client1_fx_U, client1_fx_C)
    _, fx_server_partyB = server_model_partyB(client2_fx_U, client2_fx_C)
    client1_fx = torch.cat((client1_fx_U, client1_fx_C), dim = 1)
    client2_fx = torch.cat((client2_fx_U, client2_fx_C), dim = 1)
    _, fx_server_partyAB = server_model_partyAB(client2_fx, client1_fx)
   
    loss_partyA = criterion(fx_server_partyA, Y_1).item()
    loss_partyB = criterion(fx_server_partyB, Y_1).item()
    loss_partyAB = criterion(fx_server_partyAB, Y_1).item()

    pred_partyA = fx_server_partyA.argmax(1)  
    pred_partyB = fx_server_partyB.argmax(1)  
    pred_partyAB = fx_server_partyAB.argmax(1)  

    predictions = torch.stack([pred_partyAB, pred_partyA, pred_partyB], dim=0)  # Shape: (3, batch_size)
    final_predictions, _ = torch.mode(predictions, dim=0)  

    test_correct_vote += (final_predictions == Y_1).type(torch.float).sum().item()
    test_correct_fed += (fx_server_partyAB.argmax(1) == Y_1).type(torch.float).sum().item()
    test_correct_partyB += (fx_server_partyB.argmax(1) == Y_1).type(torch.float).sum().item()
    test_correct_partyA += (fx_server_partyA.argmax(1) == Y_1).type(torch.float).sum().item()

    current = (batch_id + 1) * len(Y_1)

    if batch_id == len(test_data) - 1:
        print("---Testing Result---", file=filename)
        print(f"---epoch: {t+1}", file=filename)

        test_correct_vote_ratio = test_correct_vote/size
        test_correct_fed_ratio = test_correct_fed/size
        test_correct_partyA_ratio = test_correct_partyA / size
        test_correct_partyB_ratio = test_correct_partyB /size
        
        print(f"Vote Accuracy: {(100 * test_correct_vote_ratio):>0.1f}%",
              f"Fed Accuracy: {(100 * test_correct_fed_ratio):>0.1f}%",
              f"PartyA Accuracy: {(100 * test_correct_partyA_ratio):>0.1f}%",
              f"PartyB Accuracy: {(100 * test_correct_partyB_ratio):>0.1f}%",
              f"Test-loss-partyA: {loss_partyA:>7f}",
              f"Test-loss-partyB: {loss_partyB:>7f}",
              f"Test-loss-partyAB: {loss_partyAB:>7f}",
              F"Processing: [{current:>5d}/{size:>5d}]",file=filename)
 
        test_acc_vote.append(100 * test_correct_vote_ratio)
        test_acc_fed.append(100 * test_correct_fed_ratio)
        test_acc_partyA.append(100 * test_correct_partyA_ratio)
        test_acc_partyB.append(100 * test_correct_partyB_ratio)

        test_loss_partyA.append(loss_partyA)
        test_loss_partyB.append(loss_partyB)
        test_loss_partyAB.append(loss_partyAB)

    return test_correct_vote, test_correct_fed,test_correct_partyA,test_correct_partyB
# Test_Client Side Program
def test_client(dataloader, t):

    
    client_model_1U.eval()
    client_model_1C.eval()
    client_model_2U.eval()
    client_model_2C.eval()

    test_correct_vote = 0
    test_correct_fed = 0
    test_correct_partyA = 0
    test_correct_partyB = 0

    #size = len(dataloader.dataset)
    size = len(dataloader) * args.batch_size

    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X_1, X_2 = split_data(args.dataset, X)

        if args.dataset == 'utkface' or args.dataset == 'celeba':
            Y_1 = target[0].to(device)

        else:
            Y_1 = target.to(device)

        # client1--train and update
        fx1_U = client_model_1U(X_1)
        fx1_C = client_model_1C(X_1)
        fx2_U = client_model_2U(X_2)
        fx2_C = client_model_2C(X_2)

        client1_fx_U = (fx1_U).clone().detach().requires_grad_(True)
        client1_fx_C = (fx1_C).clone().detach().requires_grad_(True)
        client2_fx_U = (fx2_U).clone().detach().requires_grad_(True)
        client2_fx_C = (fx2_C).clone().detach().requires_grad_(True)

        # Sending activations to server and receiving gradients from server
        test_correct_vote, test_correct_fed, test_correct_partyA, test_correct_partyB = test_server(client1_fx_U, client1_fx_C, 
                              client2_fx_U, client2_fx_C,
                              Y_1, batch_id, test_correct_vote, test_correct_fed, size,
                              test_correct_partyA, test_correct_partyB)

def test_two_partys_zero(dataloader, t):
    client_model_1U.eval()
    client_model_1C.eval()
    client_model_2U.eval()
    client_model_2C.eval()
   
    correct_vote_basedOnPartyA = [0] * 7 
    correct_fed_basedOnPartyA = [0] * 7
    correct_partyB_basedOnPartyA = [0] * 7
    correct_partyA_basedOnPartyA = [0] * 7
    correct_vote_basedOnPartyB = [0] * 7
    correct_fed_basedOnPartyB = [0] * 7
    correct_partyB_basedOnPartyB = [0] * 7
    correct_partyA_basedOnPartyB = [0] * 7
    
    size = len(dataloader) * args.batch_size

    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X_1, X_2 = split_data(args.dataset, X)
        
        X_1_reserved = X_1.clone()
        X_2_reserved = X_2.clone()

        if args.dataset == 'utkface' or args.dataset == 'celeba':
            Y_1 = target[0].to(device)
        else:
            Y_1 = target.to(device)

        fx2_U = client_model_2U(X_2)
        fx2_C = client_model_2C(X_2)
        client2_fx_U = (fx2_U).clone().detach().requires_grad_(True)
        client2_fx_C = (fx2_C).clone().detach().requires_grad_(True)

        fx1_U = client_model_1U(X_1)
        fx1_C = client_model_1C(X_1)
        client1_fx_U = (fx1_U).clone().detach().requires_grad_(True)
        client1_fx_C = (fx1_C).clone().detach().requires_grad_(True)

        
        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        partyA_zeros = []  

        for i, missing_ratioA in enumerate(missing_ratios):
            
            if args.dataset == 'cifar10' or args.dataset == "imagenet" or args.dataset == 'utkface':
            
                num_zeros = int(X.shape[-1]/2)
                shape = list(range(int(num_zeros * (1 - missing_ratioA)), num_zeros)) 
                index = torch.tensor(shape).to(device)
                if index.numel() > 0:
                    X_1_blank = X_1.index_fill(3, index, 0)
                else:
                    X_1_blank = X_1
                    
            
            else:
                if missing_ratioA == 0:
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
                        Missing_item_A = missing_ratio_to_item_A.get(missing_ratioA, None)  
                        
                        X_1[:,-Missing_item_A:] = 0
                        X_1_blank = X_1
                        X_1 = X_1_reserved
                       
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
                        
                        X_1[:,-Missing_item_A:] = 0
                        X_1_blank = X_1
                        X_1 = X_1_reserved
                        
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

                        X_1[:,-Missing_item_A:] = 0
                        X_1_blank = X_1
                        X_1 = X_1_reserved

            fx1_blank_U = client_model_1U(X_1_blank)
            fx1_blank_C = client_model_1C(X_1_blank)

            partyA_zero_U = (fx1_blank_U).clone().detach().requires_grad_(True)
            partyA_zero_C = (fx1_blank_C).clone().detach().requires_grad_(True)

            partyA_zeros.append((partyA_zero_U,partyA_zero_C))

        partyB_zeros = []

        for i, missing_ratioB in enumerate(missing_ratios):
            
            if args.dataset == 'cifar10' or args.dataset == "imagenet" or args.dataset == 'utkface':
            
                num_zeros = int(X.shape[-1]/2)
                shape = list(range(num_zeros, int(num_zeros + num_zeros * missing_ratioB))) 
                index = torch.tensor(shape).to(device)
                if index.numel() > 0:
                    X_2_blank = X_2.index_fill(3, index, 0)
                else:
                    X_2_blank = X_2
                    
            else:
                if missing_ratioB == 0:
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
                        Missing_item_B = missing_ratio_to_item_B.get(missing_ratioB, None)  

                        X_2[:,-Missing_item_B:] = 0
                        X_2_blank = X_2
                        X_2 = X_2_reserved
                        
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

                        X_2[:,-Missing_item_B:] = 0
                        X_2_blank = X_2
                        X_2 = X_2_reserved
                        
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
                        
                        X_2[:,-Missing_item_B:] = 0
                        X_2_blank = X_2
                        X_2 = X_2_reserved
                        
                        
            fx2_blank_U = client_model_2U(X_2_blank)
            fx2_blank_C = client_model_2C(X_2_blank)
            partyB_zero_U = (fx2_blank_U).clone().detach().requires_grad_(True)
            partyB_zero_C = (fx2_blank_C).clone().detach().requires_grad_(True)
            partyB_zeros.append((partyB_zero_U,partyB_zero_C))

        # Sending activations to server and receiving gradients from server
        correct_vote_basedOnPartyA, correct_fed_basedOnPartyA,\
                                correct_partyB_basedOnPartyA , correct_partyA_basedOnPartyA,\
                                correct_vote_basedOnPartyB, correct_fed_basedOnPartyB,\
                                correct_partyB_basedOnPartyB , correct_partyA_basedOnPartyB = test_server_two_parties_zero(client1_fx_U, client1_fx_C, client2_fx_U, client2_fx_C, 
                                partyA_zeros, partyB_zeros,
                                correct_vote_basedOnPartyA, correct_fed_basedOnPartyA,
                                correct_partyB_basedOnPartyA , correct_partyA_basedOnPartyA,
                                correct_vote_basedOnPartyB, correct_fed_basedOnPartyB,
                                correct_partyB_basedOnPartyB , correct_partyA_basedOnPartyB,
                                Y_1, batch_id, size)


def test_server_two_parties_zero(client1_fx_U, client1_fx_C, client2_fx_U, client2_fx_C, 
                                partyA_zeros, partyB_zeros,
                                correct_vote_basedOnPartyA, correct_fed_basedOnPartyA,
                                correct_partyB_basedOnPartyA , correct_partyA_basedOnPartyA,
                                correct_vote_basedOnPartyB, correct_fed_basedOnPartyB,
                                correct_partyB_basedOnPartyB , correct_partyA_basedOnPartyB,
                                Y_1, batch_id, size):

    server_model_partyA.eval()
    server_model_partyB.eval()
    server_model_partyAB.eval()


    correct_vote_basedOnPartyA = correct_vote_basedOnPartyA
    correct_fed_basedOnPartyA = correct_fed_basedOnPartyA
    correct_partyB_basedOnPartyA = correct_partyB_basedOnPartyA
    correct_partyA_basedOnPartyA = correct_partyA_basedOnPartyA

    correct_vote_basedOnPartyB = correct_vote_basedOnPartyB
    correct_fed_basedOnPartyB = correct_fed_basedOnPartyB
    correct_partyB_basedOnPartyB = correct_partyB_basedOnPartyB
    correct_partyA_basedOnPartyB = correct_partyA_basedOnPartyB


    loss_partyA_basedOnPartyA = [0] * len(partyB_zeros)
    loss_partyB_basedOnPartyA = [0] * len(partyB_zeros)
    loss_partyAB_basedOnPartyA = [0] * len(partyB_zeros)

    for i, partyB_zero in enumerate(partyB_zeros):

        optimizer_server_partyA.zero_grad()
        optimizer_server_partyB.zero_grad()
        optimizer_server_partyAB.zero_grad()

        _, fx_server_partyA_basedOnPartyA = server_model_partyA(client1_fx_U, client1_fx_C)
        _, fx_server_partyB_basedOnPartyA = server_model_partyB(partyB_zero[0], partyB_zero[1])
        client1_fx_basedOnPartyA = torch.cat((client1_fx_U, client1_fx_C), dim = 1)
        client2_fx_basedOnPartyA = torch.cat((partyB_zero[0], partyB_zero[1]), dim = 1)
        _, fx_server_partyAB_basedOnPartyA = server_model_partyAB(client2_fx_basedOnPartyA, client1_fx_basedOnPartyA)
   
        loss_partyA_basedOnPartyA[i] = criterion(fx_server_partyA_basedOnPartyA, Y_1).item()
        loss_partyB_basedOnPartyA[i] = criterion(fx_server_partyB_basedOnPartyA, Y_1).item()
        loss_partyAB_basedOnPartyA[i] = criterion(fx_server_partyAB_basedOnPartyA, Y_1).item()

        pred_partyA_basedOnPartyA = fx_server_partyA_basedOnPartyA.argmax(1)  
        pred_partyB_basedOnPartyA = fx_server_partyB_basedOnPartyA.argmax(1)  
        pred_partyAB_basedOnPartyA = fx_server_partyAB_basedOnPartyA.argmax(1)  

        predictions_basedOnPartyA = torch.stack([pred_partyAB_basedOnPartyA, pred_partyA_basedOnPartyA, pred_partyB_basedOnPartyA], dim=0)  
        final_predictions_basedOnPartyA, _ = torch.mode(predictions_basedOnPartyA, dim=0)  

        correct_vote_basedOnPartyA[i] += (final_predictions_basedOnPartyA == Y_1).type(torch.float).sum().item()
        correct_fed_basedOnPartyA[i] += (fx_server_partyAB_basedOnPartyA.argmax(1) == Y_1).type(torch.float).sum().item()
        correct_partyB_basedOnPartyA[i] += (fx_server_partyB_basedOnPartyA.argmax(1) == Y_1).type(torch.float).sum().item()
        correct_partyA_basedOnPartyA[i] += (fx_server_partyA_basedOnPartyA.argmax(1) == Y_1).type(torch.float).sum().item()

        current = (batch_id + 1) * len(Y_1)
    

    loss_partyA_basedOnPartyB = [0] * len(partyA_zeros) 
    loss_partyB_basedOnPartyB = [0] * len(partyA_zeros) 
    loss_partyAB_basedOnPartyB = [0] * len(partyA_zeros) 

    for i, partyA_zero in enumerate(partyA_zeros):

        optimizer_server_partyA.zero_grad()
        optimizer_server_partyB.zero_grad()
        optimizer_server_partyAB.zero_grad()

        _, fx_server_partyA_basedOnPartyB = server_model_partyA(partyA_zero[0], partyA_zero[1])
        _, fx_server_partyB_basedOnPartyB = server_model_partyB(client2_fx_U, client2_fx_C)

        client2_fx_basedOnPartyB = torch.cat((client2_fx_U, client2_fx_C), dim = 1)
        client1_fx_basedOnPartyB = torch.cat((partyA_zero[0], partyA_zero[1]), dim = 1)
        _, fx_server_partyAB_basedOnPartyB = server_model_partyAB(client2_fx_basedOnPartyB, client1_fx_basedOnPartyB)

        loss_partyA_basedOnPartyB[i] = criterion(fx_server_partyA_basedOnPartyB, Y_1).item()
        loss_partyB_basedOnPartyB[i] = criterion(fx_server_partyB_basedOnPartyB, Y_1).item()
        loss_partyAB_basedOnPartyB[i] = criterion(fx_server_partyAB_basedOnPartyB, Y_1).item()

        pred_partyA_basedOnPartyB = fx_server_partyA_basedOnPartyB.argmax(1)  
        pred_partyB_basedOnPartyB = fx_server_partyB_basedOnPartyB.argmax(1)  
        pred_partyAB_basedOnPartyB = fx_server_partyAB_basedOnPartyB.argmax(1)  

        predictions_basedOnPartyB = torch.stack([pred_partyAB_basedOnPartyB, pred_partyA_basedOnPartyB, pred_partyB_basedOnPartyB], dim=0)  
        final_predictions_basedOnPartyB, _ = torch.mode(predictions_basedOnPartyB, dim=0)  

        correct_vote_basedOnPartyB[i] += (final_predictions_basedOnPartyB == Y_1).type(torch.float).sum().item()
        correct_fed_basedOnPartyB[i] += (fx_server_partyAB_basedOnPartyB.argmax(1) == Y_1).type(torch.float).sum().item()
        correct_partyB_basedOnPartyB[i] += (fx_server_partyB_basedOnPartyB.argmax(1) == Y_1).type(torch.float).sum().item()
        correct_partyA_basedOnPartyB[i] += (fx_server_partyA_basedOnPartyB.argmax(1) == Y_1).type(torch.float).sum().item()

        current = (batch_id + 1) * len(Y_1)

    if batch_id == len(test_data) - 1:
        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        for i, missing_ratio in enumerate(missing_ratios):
            print(f"Missing ratio: {missing_ratio}", file=filename)

            correct_vote_basedOnPartyA_ratio = correct_vote_basedOnPartyA[i] / size
            correct_fed_basedOnPartyA_ratio = correct_fed_basedOnPartyA[i] / size
            correct_partyB_basedOnPartyA_ratio = correct_partyB_basedOnPartyA[i] / size
            correct_partyA_basedOnPartyA_ratio = correct_partyA_basedOnPartyA[i] / size

            correct_vote_basedOnPartyB_ratio = correct_vote_basedOnPartyB[i] / size
            correct_fed_basedOnPartyB_ratio = correct_fed_basedOnPartyB[i] / size
            correct_partyB_basedOnPartyB_ratio = correct_partyB_basedOnPartyB[i] / size
            correct_partyA_basedOnPartyB_ratio = correct_partyA_basedOnPartyB[i] / size

            correct_both_vote_ratio =  (correct_vote_basedOnPartyA_ratio + correct_vote_basedOnPartyB_ratio)/2
            correct_both_fed_ratio =  (correct_fed_basedOnPartyA_ratio + correct_fed_basedOnPartyB_ratio)/2
            correct_both_partyB_ratio =  (correct_partyB_basedOnPartyA_ratio + correct_partyB_basedOnPartyB_ratio)/2
            correct_both_partyA_ratio =  (correct_partyA_basedOnPartyA_ratio + correct_partyA_basedOnPartyB_ratio)/2

            print((
                f"----For partyA is completed and partyB is completed by zero----\n"
                f"    Test-vote-Accuracy: {(100 * correct_vote_basedOnPartyA_ratio):>0.1f}%\n"
                f"    Test-fed-Accuracy: {(100 * correct_fed_basedOnPartyA_ratio):>0.1f}%\n"
                f"    Test-partyA-Accuracy: {(100 * correct_partyA_basedOnPartyA_ratio):>0.1f}%\n"
                f"    Test-partyB-Accuracy: {(100 * correct_partyB_basedOnPartyA_ratio):>0.1f}%\n"
                f"    Test-partyA-Loss: {loss_partyA_basedOnPartyA[i]:>7f}\n"
                f"    Test-partyB-Loss: {loss_partyB_basedOnPartyA[i]:>7f}\n"
                f"    Test-partyAB-Loss: {loss_partyAB_basedOnPartyA[i]:>7f}\n"
                f"----For partyB is completed and partyA is completed by zero----\n"
                f"    Test-vote-Accuracy: {(100 * correct_vote_basedOnPartyB_ratio):>0.1f}%\n"
                f"    Test-fed-Accuracy: {(100 * correct_fed_basedOnPartyB_ratio):>0.1f}%\n"
                f"    Test-partyA-Accuracy: {(100 * correct_partyA_basedOnPartyB_ratio):>0.1f}%\n"
                f"    Test-partyB-Accuracy: {(100 * correct_partyB_basedOnPartyB_ratio):>0.1f}%\n"
                f"    Test-partyA-Loss: {loss_partyA_basedOnPartyB[i]:>7f}\n"
                f"    Test-partyB-Loss: {loss_partyB_basedOnPartyB[i]:>7f}\n"
                f"    Test-partyAB-Loss: {loss_partyAB_basedOnPartyB[i]:>7f}\n"
                f"----For the average accuracy of the two situations----\n"
                f"    Test-vote-Accuracy: {(100 * correct_both_vote_ratio):>0.1f}%\n"
                f"    Test-fed-Accuracy: {(100 * correct_both_fed_ratio):>0.1f}%\n"
                f"    Test-partyA-Accuracy: {(100 * correct_both_partyA_ratio):>0.1f}%\n"
                f"    Test-partyB-Accuracy: {(100 * correct_both_partyB_ratio):>0.1f}%\n"
                f"    Processing Progress: [{current:>5d}/{size:>5d}]\n"
            ), file=filename)

            # store the relative
            test_acc_vote_basedPartyA[i].append(100 * correct_vote_basedOnPartyA_ratio)
            test_acc_fed_basedPartyA[i].append(100 * correct_fed_basedOnPartyA_ratio)
            test_acc_partyA_basedPartyA[i].append(100 * correct_partyA_basedOnPartyA_ratio)
            test_acc_partyB_basedPartyA[i].append(100 * correct_partyB_basedOnPartyA_ratio)

            test_acc_vote_basedPartyB[i].append(100 * correct_vote_basedOnPartyB_ratio)
            test_acc_fed_basedPartyB[i].append(100 * correct_fed_basedOnPartyB_ratio)
            test_acc_partyA_basedPartyB[i].append(100 * correct_partyA_basedOnPartyB_ratio)
            test_acc_partyB_basedPartyB[i].append(100 * correct_partyB_basedOnPartyB_ratio)

            test_acc_vote_avg[i].append(100 * correct_both_vote_ratio)
            test_acc_fed_avg[i].append(100 * correct_both_fed_ratio)
            test_acc_partyA_avg[i].append(100 * correct_both_partyA_ratio)
            test_acc_partyB_avg[i].append(100 * correct_both_partyB_ratio)

            test_loss_partyA_basedOnPartyA[i].append(loss_partyA_basedOnPartyA[i]) 
            test_loss_partyB_basedOnPartyA[i].append(loss_partyB_basedOnPartyA[i]) 
            test_loss_partyAB_basedOnPartyA[i].append(loss_partyAB_basedOnPartyA[i])

            test_loss_partyA_basedOnPartyB[i].append(loss_partyA_basedOnPartyB[i]) 
            test_loss_partyB_basedOnPartyB[i].append(loss_partyB_basedOnPartyB[i]) 
            test_loss_partyAB_basedOnPartyB[i].append(loss_partyAB_basedOnPartyB[i])

    return correct_vote_basedOnPartyA, correct_fed_basedOnPartyA,\
                                correct_partyB_basedOnPartyA , correct_partyA_basedOnPartyA,\
                                correct_vote_basedOnPartyB, correct_fed_basedOnPartyB,\
                                correct_partyB_basedOnPartyB , correct_partyA_basedOnPartyB

def find_top_k_average(test_data, k=5):

    # Ensure the dataset has at least k elements
    if len(test_data) < k:
        return None

    # Sort the dataset in descending order and select the top k elements
    top_k_values = sorted(test_data, reverse=True)[:k]

    # Compute and return the mean of the top k values
    return np.mean(top_k_values)



if __name__ == '__main__':
    # Define record path
    root_path = '.'
    data_path = os.path.join(root_path, 'Dataset').replace('\\', '/')
    save_path = f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = open(f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/A_training_process.txt', 'w+')
    

    ### Load data 
    if args.dataset == 'utkface' or args.dataset == 'celeba':
              
        train_data_nondataloader, test_data, num_classes, num_classes2, channel, hidden = load_data(args.dataset, args.attributes, data_path,
                                                                                                  args.batch_size)
        
    elif args.dataset =='MIMIC' or args.dataset == 'avazu' or args.dataset == 'bank':
        train_data_nondataloader, test_data, input_dim1, input_dim2, num_classes = load_data_tabular(args.dataset, args.batch_size)
        
    elif args.dataset == "imagenet":
        train_data_nondataloader, test_data, num_classes, channel, hidden = get_tinyimagenet_bothloader(batch_size=args.batch_size, shuffle=True, seed=args.seed, 
                                                                                          device=device, classes = args.classes)

    else:
        train_data_nondataloader, test_data, num_classes, channel, hidden = gen_dataset(args.dataset, data_path, device = device)

    # Define model
    if args.dataset =='MIMIC' or args.dataset == 'avazu' or args.dataset == 'bank':
        client_model_1U, client_model_1C, client_model_2U, client_model_2C, \
            server_model_partyA, server_model_partyB, server_model_partyAB = def_tabular_model_CVT(dataset= args.dataset, level= args.level, input_dim1=input_dim1, input_dim2=input_dim2, num_classes=num_classes, num_cutlayer=args.num_cutlayer)
        
    else:

        ImageModel_cvt= ImageModel_CVT(dataset= args.dataset, hidden=hidden, num_cutlayer=args.num_cutlayer, num_classes=num_classes, device=device)
        client_model_1U, client_model_1C, client_model_2U, client_model_2C, \
            server_model_partyA, server_model_partyB, server_model_partyAB = ImageModel_cvt.GetModel()

    # criterion & optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer_client1U = torch.optim.Adam(client_model_1U.parameters(), lr=args.lr, foreach=False )
    optimizer_client1U = torch.optim.Adam(client_model_1C.parameters(), lr=args.lr, foreach=False )
    optimizer_client2U = torch.optim.Adam(client_model_2U.parameters(), lr=args.lr, foreach=False )
    optimizer_client2C = torch.optim.Adam(client_model_2C.parameters(), lr=args.lr, foreach=False )
    optimizer_server_partyA  = torch.optim.Adam(server_model_partyA.parameters(),   lr=args.lr, foreach=False ) 
    optimizer_server_partyB  = torch.optim.Adam(server_model_partyB.parameters(),   lr=args.lr, foreach=False )  
    optimizer_server_partyAB  = torch.optim.Adam(server_model_partyAB.parameters(),   lr=args.lr, foreach=False )  

    # record results
    train_A_acc = []
    train_B_acc = []
    train_Fed_acc = []

    train_loss = []

    test_acc_vote = []
    test_acc_fed = []
    test_acc_partyA = []
    test_acc_partyB = []

    test_loss_partyA = []
    test_loss_partyB = []
    test_loss_partyAB = []


    test_acc_vote_basedPartyA = [[] for _ in range(7)]
    test_acc_fed_basedPartyA = [[] for _ in range(7)]
    test_acc_partyA_basedPartyA = [[] for _ in range(7)]
    test_acc_partyB_basedPartyA = [[] for _ in range(7)]
    test_acc_vote_basedPartyB = [[] for _ in range(7)]
    test_acc_fed_basedPartyB = [[] for _ in range(7)]
    test_acc_partyA_basedPartyB = [[] for _ in range(7)]
    test_acc_partyB_basedPartyB = [[] for _ in range(7)]
    test_acc_vote_avg = [[] for _ in range(7)]
    test_acc_fed_avg = [[] for _ in range(7)]
    test_acc_partyA_avg = [[] for _ in range(7)]
    test_acc_partyB_avg = [[] for _ in range(7)]

    test_loss_partyA_basedOnPartyA = [[] for _ in range(7)]
    test_loss_partyB_basedOnPartyA = [[] for _ in range(7)]
    test_loss_partyAB_basedOnPartyA = [[] for _ in range(7)]
    test_loss_partyA_basedOnPartyB = [[] for _ in range(7)]
    test_loss_partyB_basedOnPartyB = [[] for _ in range(7)]
    test_loss_partyAB_basedOnPartyB = [[] for _ in range(7)]

    # start training
    for t in range(args.epochs):
        print(f"Epoch {t + 1}\n-------------------------------", file=filename)
        print(t)

        if args.dataset == 'cifar10' or args.dataset == "imagenet" or args.dataset == 'utkface' or args.dataset =='MIMIC' or args.dataset == 'bank' or args.dataset == 'avazu':
            train_data = get_train(train_data_nondataloader, args.batch_size, args.seed, device) 

        train_client(train_data, client_model_1U, client_model_1C,
                client_model_2U, client_model_2C,
                server_model_partyA, server_model_partyB,
                server_model_partyAB, t)
    print("Done!", file=filename)

    print("----Final result----", file=filename)
    print("----Find the Max result----", file=filename)

    print("Max--test_acc_vote: ", np.array(test_acc_vote).max(), file=filename)
    print("Max--test_acc_fed: ", np.array(test_acc_fed).max(), file=filename)
    print("Max--test_acc_partyA: ", np.array(test_acc_partyA).max(), file=filename)
    print("Max--test_acc_partyB: ", np.array(test_acc_partyB).max(), file=filename)
    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        print(f"Missing_ratio: {missing_ratio}", file=filename)
        print("Max--test_acc_vote_avg: ", np.array(test_acc_vote_avg[i]).max(), file=filename)
        print(f"where test_acc_vote_basedPartyA:"
              f"{np.array(test_acc_vote_basedPartyA[i])[np.array(test_acc_vote_avg[i]).argmax()]}\n"
              f"      test_acc_vote_basedPartyB:"
              f"{np.array(test_acc_vote_basedPartyB[i])[np.array(test_acc_vote_avg[i]).argmax()]}\n"
              f"      test_acc_fed_basedPartyA:"
              f"{np.array(test_acc_fed_basedPartyA[i])[np.array(test_acc_vote_avg[i]).argmax()]}\n"
              f"      test_acc_fed_basedPartyB:"
              f"{np.array(test_acc_fed_basedPartyB[i])[np.array(test_acc_vote_avg[i]).argmax()]}\n"
              f"      test_acc_partyA_basedPartyA:"
              f"{np.array(test_acc_partyA_basedPartyA[i])[np.array(test_acc_vote_avg[i]).argmax()]}\n"
              f"      test_acc_partyA_basedPartyB:"
              f"{np.array(test_acc_partyA_basedPartyB[i])[np.array(test_acc_vote_avg[i]).argmax()]}\n"
              f"      test_acc_partyB_basedPartyA:"
              f"{np.array(test_acc_partyB_basedPartyA[i])[np.array(test_acc_vote_avg[i]).argmax()]}\n"
              f"      test_acc_partyB_basedPartyB:"
              f"{np.array(test_acc_partyB_basedPartyB[i])[np.array(test_acc_vote_avg[i]).argmax()]}\n"
              , file=filename

              )


    print("----Final result----", file=filename)
    print("----Find the top k result----", file=filename)

    print("topk--test_acc_vote: ", find_top_k_average(test_acc_vote), file=filename)
    print("topk--test_acc_fed: ", find_top_k_average(test_acc_fed), file=filename)
    print("topk--test_acc_partyA: ", find_top_k_average(test_acc_partyA), file=filename)
    print("topk--test_acc_partyB: ", find_top_k_average(test_acc_partyB), file=filename)

    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        print(f"Missing_ratio: {missing_ratio}", file=filename)

        k = 5  
        test_acc_vote_avg_array = np.array(test_acc_vote_avg[i])
        top_k_indices = test_acc_vote_avg_array.argsort()[-k:][::-1]  

        top_k_vote_avg = test_acc_vote_avg_array[top_k_indices].mean()
        print("topk--test_acc_vote_avg: ", top_k_vote_avg, file=filename)

        topk_vote_basedPartyA = np.array(test_acc_vote_basedPartyA[i])[top_k_indices].mean()
        topk_vote_basedPartyB = np.array(test_acc_vote_basedPartyB[i])[top_k_indices].mean()
        topk_fed_basedPartyA = np.array(test_acc_fed_basedPartyA[i])[top_k_indices].mean()
        topk_fed_basedPartyB = np.array(test_acc_fed_basedPartyB[i])[top_k_indices].mean()
        topk_partyA_basedPartyA = np.array(test_acc_partyA_basedPartyA[i])[top_k_indices].mean()
        topk_partyA_basedPartyB = np.array(test_acc_partyA_basedPartyB[i])[top_k_indices].mean()
        topk_partyB_basedPartyA = np.array(test_acc_partyB_basedPartyA[i])[top_k_indices].mean()
        topk_partyB_basedPartyB = np.array(test_acc_partyB_basedPartyB[i])[top_k_indices].mean()

        print(f"where topk_test_acc_vote_basedPartyA: {topk_vote_basedPartyA}\n"
              f"      topk_test_acc_vote_basedPartyB: {topk_vote_basedPartyB}\n"
              f"      topk_test_acc_fed_basedPartyA: {topk_fed_basedPartyA}\n"
              f"      topk_test_acc_fed_basedPartyB: {topk_fed_basedPartyB}\n"
              f"      topk_test_acc_partyA_basedPartyA: {topk_partyA_basedPartyA}\n"
              f"      topk_test_acc_partyA_basedPartyB: {topk_partyA_basedPartyB}\n"
              f"      topk_test_acc_partyB_basedPartyA: {topk_partyB_basedPartyA}\n"
              f"      topk_test_acc_partyB_basedPartyB: {topk_partyB_basedPartyB}\n",
              file=filename)
        
        print('-----------------Seperate Best------------------------', file=filename)
        print("topk--test_acc_partyA_basedPartyA: ", find_top_k_average(test_acc_partyA_basedPartyA[i]), file=filename)
        print("topk--test_acc_partyA_basedPartyB: ", find_top_k_average(test_acc_partyA_basedPartyB[i]), file=filename)
        print("topk--test_acc_partyB_basedPartyA: ", find_top_k_average(test_acc_partyB_basedPartyA[i]), file=filename)
        print("topk--test_acc_partyB_basedPartyB: ", find_top_k_average(test_acc_partyB_basedPartyB[i]), file=filename)
        print('------------------------------------------------------', file=filename)

    #plt
    x = np.arange(len(train_A_acc))
    plt.plot(x, train_A_acc, label='train_A_acc')
    plt.plot(x, train_B_acc, label='train_B_acc')
    plt.plot(x, train_Fed_acc, label='train_Fed_acc')
    plt.xlabel('epoch',   fontsize=19)
    plt.ylabel('ACC',   fontsize=19)
    plt.title(f'training_{args.dataset}_overlap{args.num_overlap}',   fontsize=20)
    plt.legend()
    plt.savefig(f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/A_train_ACC.png')
    plt.close()

    x = np.arange(len(test_acc_vote))
    plt.plot(x, test_acc_vote,  label='test_acc_vote')
    plt.plot(x, test_acc_fed, label='test_acc_fed')
    plt.plot(x, test_acc_partyA, label='test_acc_partyA')
    plt.plot(x, test_acc_partyB, label='test_acc_partyB')

    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        plt.plot(x, test_acc_vote_avg[i],  label='test_acc_vote_avg')

    plt.xlabel('epoch',   fontsize=19)
    plt.ylabel('ACC',   fontsize=19)
    plt.title(f'testing_{args.dataset}_overlap{args.num_overlap}',   fontsize=20)
    plt.legend()
    plt.savefig(f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/A_test_ACC.png')
    plt.close()


    x = np.arange(len(train_loss))
    plt.plot(x, train_loss, label='train_loss')
    plt.xlabel('epoch',   fontsize=19)
    plt.ylabel('Loss',   fontsize=19)
    plt.title(f'training_{args.dataset}_overlap{args.num_overlap}',   fontsize=20)
    plt.legend()
    plt.savefig(f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/A_train_Loss.png')
    plt.close()
    
    x = np.arange(len(test_loss_partyA))
    plt.plot(x, test_loss_partyA,  label='test_loss_partyA')
    plt.plot(x, test_loss_partyB, label='test_loss_partyB')
    plt.plot(x, test_loss_partyAB, label='test_loss_partyAB')

    # save all the results of testing
    np.save(f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/test_acc_vote.npy', test_acc_vote) 
    np.save(f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/test_acc_fed.npy', test_acc_fed) 
    np.save(f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/test_acc_partyA.npy', test_acc_partyA) 
    np.save(f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/test_acc_partyB.npy', test_acc_partyB) 
    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        np.save(f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/test_acc_vote_basedPartyA_Missing{missing_ratio}.npy', test_acc_vote_basedPartyA[i]) 
        np.save(f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/test_acc_vote_basedPartyB_Missing{missing_ratio}.npy', test_acc_vote_basedPartyB[i]) 
        np.save(f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/test_acc_fed_basedPartyA_Missing{missing_ratio}.npy', test_acc_fed_basedPartyA[i]) 
        np.save(f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/test_acc_fed_basedPartyA_Missing{missing_ratio}.npy', test_acc_fed_basedPartyB[i]) 
        np.save(f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/test_acc_partyA_basedPartyA_Missing{missing_ratio}.npy', test_acc_partyA_basedPartyA[i]) 
        np.save(f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/test_acc_partyA_basedPartyB_Missing{missing_ratio}.npy', test_acc_partyA_basedPartyB[i]) 
        np.save(f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/test_acc_partyB_basedPartyA_Missing{missing_ratio}.npy', test_acc_partyB_basedPartyA[i]) 
        np.save(f'Result/Results_FedCVT/Results_batchsize{args.batch_size}/learningRate{args.lr}/sharpenT{args.tem}/{args.dataset}/Seed{args.seed}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/nonoverlap1_ratio{args.nonoverlap1_ratio}/overlap{args.num_overlap*args.num_batch}-l1{args.CVTL1}-l2{args.CVTL2}-l3{args.CVTL3}-l4{args.CVTL4}-l5{args.CVTL5}/test_acc_partyB_basedPartyB_Missing{missing_ratio}.npy', test_acc_partyB_basedPartyB[i]) 

