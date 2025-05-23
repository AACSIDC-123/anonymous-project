import os
import copy
from torch import nn
import numpy as np
from models import *
from utils import *
from parse import args
import matplotlib.pyplot as plt
from Decoder.decoder_image import *

# Define A random_seed
set_random_seed(args.seed)
best_correct_ratio = 0.0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
device = torch.device(f"cuda:0") 

# Train_Server Side Program
def train_server_overlap(client1_fx, client2_fx, Y_1, t, batch_id, correct, size):
    server_model.train()
    correct = correct
    global train_overlap_acc 
    global train_overlap_loss

    _, server_fx_avg = server_model(client1_fx, client2_fx)
    loss_ce = criterion(server_fx_avg, Y_1) 
    loss = loss_ce 
    
    # backward
    optimizer_server.zero_grad()
    loss.backward()
    dfx1 = client1_fx.grad.clone().detach().to(device)
    dfx2 = client2_fx.grad.clone().detach().to(device)
    optimizer_server.step()
    
    #acc
    correct += (server_fx_avg.argmax(1) == Y_1).type(torch.float).sum().item()

    if batch_id == args.num_batch-1:

        save_path3 = f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/server.pth'
        torch.save(server_model.state_dict(), save_path3)
        
        correct_train = correct / (args.num_overlap * args.num_batch)
        loss, current = loss.item(), (batch_id + 1) *args.batch_size
        print(f"train-overlap-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%", file=filename)
        train_overlap_acc.append(100 * correct_train)
        train_overlap_loss.append(loss)

    return dfx1, dfx2,  correct

# Train_Client Side Program
def train_client(dataloader, client_model_1, client_model_2, t):
    client_model_1.train()
    client_model_2.train()

    correct = 0
    size = len(dataloader)*args.batch_size

    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X_1, X_2 = split_data(args.dataset, X)
        
        if args.dataset == 'utkface' or args.dataset == 'celeba':
            Y_1 = target[0].to(device)
            Y_2 = target[1].view(-1, 1).to(device)
        else:
            Y_1 = target.to(device)

        X_1_overlap     = X_1[:args.num_overlap]
        X_2_overlap     = X_2[:args.num_overlap]
        X_2_non_overlap = X_1[args.num_overlap:int((args.batch_size - args.num_overlap)*args.nonoverlap1_ratio + args.num_overlap)]
        X_1_non_overlap = X_2[int((args.batch_size - args.num_overlap)*(args.nonoverlap1_ratio) + args.num_overlap):args.batch_size]
        
        Y_overlap = Y_1[:args.num_overlap]
        Y_non_overlap1 = Y_1[args.num_overlap:int((args.batch_size - args.num_overlap)*args.nonoverlap1_ratio + args.num_overlap)]
        Y_non_overlap2 = Y_1[int((args.batch_size - args.num_overlap)*(args.nonoverlap1_ratio) + args.num_overlap):args.batch_size]

        # client1--train and update
        if args.overlap == 'True':

            #E
            fx1 = client_model_1(X_1_overlap)
            fx2 = client_model_2(X_2_overlap)

            # Sending activations to server and receiving gradients from server
            client1_fx = (fx1).clone().detach().requires_grad_(True)
            client2_fx = (fx2).clone().detach().requires_grad_(True)
            g_fx1, g_fx2, correct = train_server_overlap(client1_fx, client2_fx,  Y_overlap, t, batch_id, correct, size)
            
            #backward
            optimizer_client1.zero_grad()
            optimizer_client2.zero_grad() 
            (fx1).backward(g_fx1)
            (fx2).backward(g_fx2)
            optimizer_client1.step()
            optimizer_client2.step()
        
        if batch_id == args.num_batch-1:
            save_path1 = f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client1.pth'
            save_path2 = f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client2.pth'
            torch.save(client_model_1.state_dict(), save_path1)
            torch.save(client_model_2.state_dict(), save_path2)
        # break        
        if batch_id >= args.num_batch-1:
            break
        

# test_server Side Program
def test_server(client1_fx, client2_fx, 
                partyB_zeros, partyA_zeros,
                Y_1, batch_id,
                correct_normal, correct_basedOnPartyA_cases, 
                correct_basedOnPartyB_cases,size):
    
    global best_correct_ratio
    
    server_model.eval()
    correct_normal = correct_normal
    # two lists
    correct_basedOnPartyA_cases = correct_basedOnPartyA_cases
    correct_basedOnPartyB_cases = correct_basedOnPartyB_cases

    # train and update
    optimizer_server.zero_grad()
    _, fx_server_normal = server_model(client1_fx, client2_fx)
    loss_normal = criterion(fx_server_normal, Y_1).item()
    correct_normal += (fx_server_normal.argmax(1) == Y_1).type(torch.float).sum().item()
    correct_normal_ratio = correct_normal / size

    # A is completed and B is completed by zero
    loss_basedOnPartyA = [0] * len(partyB_zeros) 
    for i, partyB_zero in enumerate(partyB_zeros):
        optimizer_server.zero_grad()
        _, fx_server_basedOnPartyA = server_model(client1_fx, partyB_zero)
        loss_basedOnPartyA[i] = criterion(fx_server_basedOnPartyA, Y_1).item()
        correct_basedOnPartyA_cases[i] += (fx_server_basedOnPartyA.argmax(1) == Y_1).type(torch.float).sum().item()
    

    # B is completed and A is completed by zero
    loss_basedOnPartyB = [0] * len(partyA_zeros) 
    for i, partyA_zero in enumerate(partyA_zeros):
        optimizer_server.zero_grad()
        _, fx_server_basedOnPartyB = server_model(partyA_zero, client2_fx)
        loss_basedOnPartyB[i] = criterion(fx_server_basedOnPartyB, Y_1).item()
        correct_basedOnPartyB_cases[i] += (fx_server_basedOnPartyB.argmax(1) == Y_1).type(torch.float).sum().item()
  

    current = (batch_id + 1) * len(Y_1)

    if batch_id == len(test_data) - 1:

        print("---Normal Case---",file=filename)
        print(f"Test-loss: {loss_normal:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_normal_ratio):>0.1f}%",
              file=filename)
        test_normal_acc.append(100 * correct_normal_ratio)
        test_normal_loss.append(loss_normal)
        
        if correct_normal_ratio > best_correct_ratio:
            best_correct_ratio = correct_normal_ratio  
            save_path1_best = f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client1_best.pth'
            save_path2_best = f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client2_best.pth'
            save_path3_best = f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/server_best.pth'
            torch.save(client_model_1.state_dict(), save_path1_best)
            torch.save(client_model_2.state_dict(), save_path2_best)
            torch.save(server_model.state_dict(), save_path3_best)

        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        print("---BasedOnPartyA Case---",file=filename)
        for i, missing_ratio in enumerate(missing_ratios):
            correct_basedOnPartyA_ratio = (correct_basedOnPartyA_cases[i] / size)
            print(f"Missing ratio: {missing_ratio}", file=filename)

            print(f"Test-loss: {loss_basedOnPartyA[i]:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_basedOnPartyA_ratio):>0.1f}%",
                  file=filename)
            test_basedOnPartyA_acc[i].append(100 * correct_basedOnPartyA_ratio)
            test_basedOnPartyA_loss[i].append(loss_basedOnPartyA[i])

        print("---BasedOnPartyB Case---",file=filename)
        for i, missing_ratio in enumerate(missing_ratios):
            correct_basedOnPartyB_ratio = (correct_basedOnPartyB_cases[i] / size)
            print(f"Missing ratio: {missing_ratio}", file=filename)

            print(f"Test-loss: {loss_basedOnPartyB[i]:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_basedOnPartyB_ratio):>0.1f}%",
                  file=filename)
            test_basedOnPartyB_acc[i].append(100 * correct_basedOnPartyB_ratio)
            test_basedOnPartyB_loss[i].append(loss_basedOnPartyB[i])
        
        correct_sum = [0] * 7
        print("---Missing Average Case---",file=filename)
        for i, missing_ratio in enumerate(missing_ratios):
            correct_sum[i] = correct_basedOnPartyA_cases[i] + correct_basedOnPartyB_cases[i]
            correct_average_ratio = (correct_sum[i] / size)/2
            print(f"Missing ratio: {missing_ratio}", file=filename)

            print(f"Accuracy: {(100 * correct_average_ratio):>0.1f}%",
                  file=filename)
            test_average_acc[i].append(100 * correct_average_ratio)

    return correct_normal, correct_basedOnPartyA_cases, correct_basedOnPartyB_cases


# Test_Client Side Program
def test_client(dataloader, client_model_1, client_model_2, t):
    client_model_1.eval()
    client_model_2.eval()

    correct_normal = 0
    correct_basedOnPartyA_cases = [0] * 7
    correct_basedOnPartyB_cases = [0] * 7

    size = len(dataloader) * args.batch_size

    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X_1, X_2 = split_data(args.dataset, X)

        if args.dataset == 'utkface' or args.dataset == 'celeba':
            Y_1 = target[0].to(device)

        else:
            Y_1 = target.to(device)

        # normal situation
        fx1 = client_model_1(X_1)
        fx2 = client_model_2(X_2)

        client1_fx = (fx1).clone().detach().requires_grad_(True)
        client2_fx = (fx2).clone().detach().requires_grad_(True)

        # R_missing situation
        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        partyA_zeros = []  
        partyB_zeros = []

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

            fx2_blank = client_model_2(X_2_blank)
            partyB_zero = (fx2_blank).clone().detach().requires_grad_(True)
            partyB_zeros.append(partyB_zero)

            if args.dataset == 'cifar10' or args.dataset == "imagenet" or args.dataset == 'utkface':
            
                # B is completed and A is completed by zeros
                num_zeros = int(X.shape[-1]/2)
                shape = list(range(int(num_zeros * (1 - missing_ratio)), num_zeros )) 
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
        
            fx1_blank = client_model_1(X_1_blank)
            partyA_zero = (fx1_blank).clone().detach().requires_grad_(True)
            partyA_zeros.append(partyA_zero)

        # Sending activations to server and receiving gradients from server
        correct_normal, correct_basedOnPartyA_cases,\
              correct_basedOnPartyB_cases = test_server(client1_fx, client2_fx, 
                                                  partyB_zeros, partyA_zeros,
                                                  Y_1, batch_id,
                                                 correct_normal, correct_basedOnPartyA_cases, 
                                                 correct_basedOnPartyB_cases,size)


if __name__ == '__main__':
    # Define record path
    root_path = '.'
    data_path = os.path.join(root_path, 'Dataset').replace('\\', '/')
    save_path = f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = open(f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/A_training_process.txt', 'w+')
    
    if args.dataset == 'utkface' or args.dataset == 'celeba':     
        train_data_nondataloader, test_data, num_classes, num_classes2, channel, hidden = load_data(args.dataset, args.attributes, data_path, args.batch_size)

    elif args.dataset =='MIMIC' or args.dataset == 'avazu' or args.dataset == 'bank':
        train_data_nondataloader, test_data, input_dim1, input_dim2, num_classes = load_data_tabular(args.dataset, args.batch_size)

    elif args.dataset == "imagenet":
        train_data_nondataloader, test_data, num_classes, channel, hidden = get_tinyimagenet_bothloader(batch_size=args.batch_size, shuffle=True, seed=args.seed, device=device, classes = args.classes)
        
    else:   
        train_data_nondataloader, test_data, num_classes, channel, hidden = gen_dataset(args.dataset, data_path, device = device)

    # Define model
    if args.dataset =='MIMIC' or args.dataset =='avazu' or args.dataset =='bank':
        client_model_1, client_model_2, server_model = def_tabular_model(dataset= args.dataset, level= args.level, input_dim1=input_dim1, input_dim2=input_dim2, num_classes=num_classes, num_cutlayer=args.num_cutlayer, mode='cat')
    else:
        ImageModel1 = ImageModel(dataset= args.dataset, hidden=hidden, num_cutlayer=args.num_cutlayer, num_classes=num_classes, mode='cat', device = device)
        client_model_1, client_model_2, server_model = ImageModel1.GetModel()

    # criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_client1 = torch.optim.Adam(client_model_1.parameters(), lr=args.lr, foreach=False )
    optimizer_client2 = torch.optim.Adam(client_model_2.parameters(), lr=args.lr, foreach=False )
    optimizer_server  = torch.optim.Adam(server_model.parameters(),   lr=args.lr, foreach=False )  

    # record results
    train_overlap_loss = []
    train_overlap_acc = []    
    test_normal_loss = []
    test_normal_acc = []
    test_basedOnPartyA_acc = [[] for _ in range(7)] 
    test_basedOnPartyA_loss = [[] for _ in range(7)] 
    test_basedOnPartyB_acc = [[] for _ in range(7)] 
    test_basedOnPartyB_loss = [[] for _ in range(7)] 
    test_average_acc = [[] for _ in range(7)] 

    # start training
    for t in range(args.epochs):
        print(f"Epoch {t + 1}\n-------------------------------", file=filename)
        print(t)

        if args.dataset == 'cifar10' or args.dataset == "imagenet" or args.dataset == 'utkface' or args.dataset =='MIMIC' or args.dataset == 'bank' or args.dataset == 'avazu':
            train_data = get_train(train_data_nondataloader, args.batch_size, args.seed, device) 
            
        train_client(train_data, client_model_1, client_model_2, t)
        test_client(test_data, client_model_1, client_model_2, t)
    print("Done!", file=filename)
    
    print(f"train_overlap_acc: {train_overlap_acc}")
    print(f"train_overlap_loss: {train_overlap_loss}")
    print("-----Max result------", file=filename)
    print("max--test_normal_acc!", np.array(test_normal_acc).max(), file=filename)
    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        print(f"Missing_ratio: {missing_ratio}", file=filename)
        print("max--test_average_acc!", np.array(test_average_acc[i]).max(), file=filename)
        print("max--test_basedOnPartyA_acc!", np.array(test_basedOnPartyA_acc[i]).max(), file=filename)
        print("max--test_basedOnPartyB_acc!", np.array(test_basedOnPartyB_acc[i]).max(), file=filename)

    k = 5 

    max_k_test_average_acc = [[] for _ in range(7)]
    max_k_test_basedOnPartyA_acc = [[] for _ in range(7)]
    max_k_test_basedOnPartyB_acc = [[] for _ in range(7)]

    max_k_test_normal_acc = np.mean(np.sort(test_normal_acc)[-k:])
    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        max_k_test_average_acc[i] = np.mean(np.sort(test_average_acc[i])[-k:])
        max_k_test_basedOnPartyA_acc[i] = np.mean(np.sort(test_basedOnPartyA_acc[i])[-k:])
        max_k_test_basedOnPartyB_acc[i] = np.mean(np.sort(test_basedOnPartyB_acc[i])[-k:])
    
    print(f"-----Top{k} result------", file=filename)
    print(f"top{k}avg--test_normal_acc: {max_k_test_normal_acc}", file=filename)
    
    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        print(f"Missing_ratio: {missing_ratio}", file=filename)
        print(f"top{k}avg--test_average_acc: {max_k_test_average_acc[i]}", file=filename)
        print(f"top{k}avg--test_basedOnPartyA_acc: {max_k_test_basedOnPartyA_acc[i]}", file=filename)
        print(f"top{k}avg--test_basedOnPartyB_acc: {max_k_test_basedOnPartyB_acc[i]}", file=filename)

    #save 
    np.save(f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/train_overlap_acc.npy',train_overlap_acc) 
    np.save(f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/train_overlap_loss.npy',train_overlap_loss)
    np.save(f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/test_normal_acc.npy',test_normal_acc) 
    np.save(f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/test_normal_loss.npy',test_normal_loss) 

    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        np.save(f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/test_basedOnPartyA_acc_Missing{missing_ratio}.npy',   test_basedOnPartyA_acc[i]) 
        np.save(f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/test_basedOnPartyA_loss_Missing{missing_ratio}.npy',  test_basedOnPartyA_loss[i]) 
        np.save(f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/test_basedOnPartyB_acc_Missing{missing_ratio}.npy',   test_basedOnPartyB_acc[i]) 
        np.save(f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/test_basedOnPartyB_loss_Missing{missing_ratio}.npy',  test_basedOnPartyB_loss[i])  
        np.save(f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/test_average_acc_Missing{missing_ratio}.npy',  test_average_acc[i])  
    
    #plt
    x = np.arange(len(train_overlap_acc))
    plt.plot(x, train_overlap_acc, label='train_overlap_acc')
    plt.xlabel('epoch',   fontsize=19)
    plt.ylabel('ACC',   fontsize=19)
    plt.title(f'Train_acc_{args.dataset}_sample{args.num_overlap}',   fontsize=20)
    plt.legend()
    plt.savefig(f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/Train_Accuracy.png')
    plt.close()

    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        x = np.arange(len(test_normal_acc))
        plt.plot(x, test_normal_acc,  label='test_normal_acc')
        plt.plot(x, test_average_acc[i],  label='test_average_acc')
        plt.plot(x, test_basedOnPartyA_acc[i],  label='test_basedOnPartyA_acc')
        plt.plot(x, test_basedOnPartyB_acc[i],  label='test_basedOnPartyB_acc')
        plt.xlabel('epoch',   fontsize=19)
        plt.ylabel('ACC',   fontsize=19)
        plt.title(f'Test_acc_{args.dataset}_sample{args.num_overlap}_Missing{missing_ratio}',   fontsize=20)
        plt.legend()
        plt.savefig(f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/Test_Acc_Missing{missing_ratio}.png')
        plt.close()
        
    x = np.arange(len(train_overlap_loss))
    plt.plot(x, train_overlap_loss, label='train_overlap_loss')
    plt.xlabel('epoch',   fontsize=19)
    plt.ylabel('Loss',   fontsize=19)
    plt.title(f'Train_loss_{args.dataset}_sample{args.num_overlap}',   fontsize=20)
    plt.legend()
    plt.savefig(f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/Train_Loss.png')
    plt.close()

    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        plt.plot(x, test_normal_loss, label='test_normal_loss')
        plt.plot(x, test_basedOnPartyA_loss[i], label='test_basedOnPartyA_loss')
        plt.plot(x, test_basedOnPartyB_loss[i], label='test_basedOnPartyB_loss')
        plt.xlabel('epoch',   fontsize=19)
        plt.ylabel('Loss',   fontsize=19)
        plt.title(f'Test_loss_{args.dataset}_sample{args.num_overlap}_Missing{missing_ratio}',   fontsize=20)
        plt.legend()
        plt.savefig(f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/Test_Loss_Missing{missing_ratio}.png')
        plt.close()






