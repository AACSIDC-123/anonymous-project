import os
import copy
import sys
from torch import nn
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
    
# Train_Server Side Program
def train_server_overlap(client1_fx, client2_fx, client3_fx, client4_fx, server_model, Y_1, t, batch_id, train_correct, size):
    server_model.train()
    train_correct = train_correct
    
    global train_overlap_acc 
    global train_overlap_loss
    
    _, server_fx_avg = server_model(client1_fx, client2_fx, client3_fx, client4_fx)
    loss_ce = criterion(server_fx_avg, Y_1) 
    loss = loss_ce 
    
    # backward
    optimizer_server.zero_grad()
    loss.backward()
    dfx1 = client1_fx.grad.clone().detach().to(device)
    dfx2 = client2_fx.grad.clone().detach().to(device)
    dfx3 = client3_fx.grad.clone().detach().to(device)
    dfx4 = client4_fx.grad.clone().detach().to(device)
    optimizer_server.step()
    
    #acc
    train_correct += (server_fx_avg.argmax(1) == Y_1).type(torch.float).sum().item()
    

    if batch_id == args.num_batch-1:
        correct_train = train_correct / (args.num_overlap * args.num_batch)
        loss, current = loss.item(), (batch_id + 1) *args.batch_size
        print(f"train-overlap-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%", file=filename)
        train_overlap_acc.append(100 * correct_train)
        train_overlap_loss.append(loss)
        
    return dfx1, dfx2, dfx3, dfx4, train_correct


def train_client(dataloader, client_model_1, client_model_2, client_model_3, client_model_4, server_model, t):
    
    client_model_1.train()
    client_model_2.train()
    client_model_3.train()
    client_model_4.train()
    
    train_correct = 0
    size = len(dataloader)*args.batch_size
    
    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X_1, X_2, X_3, X_4 = split_data_4clients(args.dataset, X)
        
        if args.dataset == 'utkface' or args.dataset == 'celeba':
            Y_1 = target[0].to(device)
            Y_2 = target[1].view(-1, 1).to(device)
        else:
            Y_1 = target.to(device)
            
        X_1_overlap = X_1[:args.num_overlap]
        X_2_overlap = X_2[:args.num_overlap]
        X_3_overlap = X_3[:args.num_overlap]
        X_4_overlap = X_4[:args.num_overlap]
        
        Y_overlap = Y_1[:args.num_overlap]
        
        if args.overlap == 'True':
            
            fx1 = client_model_1(X_1_overlap)
            fx2 = client_model_2(X_2_overlap)
            fx3 = client_model_3(X_3_overlap)
            fx4 = client_model_4(X_4_overlap)
            
            
            client1_fx = (fx1).clone().detach().requires_grad_(True)
            client2_fx = (fx2).clone().detach().requires_grad_(True)
            client3_fx = (fx3).clone().detach().requires_grad_(True)
            client4_fx = (fx4).clone().detach().requires_grad_(True)
            
            g_fx1, g_fx2, g_fx3, g_fx4, train_correct = train_server_overlap(client1_fx, client2_fx, client3_fx, client4_fx, server_model, Y_overlap, t, batch_id, train_correct, size)
            
            #backward
            optimizer_client1.zero_grad()
            optimizer_client2.zero_grad()
            optimizer_client3.zero_grad()
            optimizer_client4.zero_grad()  
            (fx1).backward(g_fx1)
            (fx2).backward(g_fx2)
            (fx3).backward(g_fx3)
            (fx4).backward(g_fx4)
            optimizer_client1.step()
            optimizer_client2.step()
            optimizer_client3.step()
            optimizer_client4.step()
            
   
        if batch_id >= args.num_batch-1:
            break
        

def test_server(dataloader, clients_embeddings, server_model, Y_1, batch_id, size, test_acc, test_loss, test_correct, t):
    
    global best_correct_ratio
    
    server_model.eval()
    
    # [[
    #  [0, 0, 0, 0, 0, 0, 0],  # party1 is partially missing
    #  [0, 0, 0, 0, 0, 0, 0],  # party2 is partially missing
    #  [0, 0, 0, 0, 0, 0, 0],  # party3 is partially missing
    #  [0, 0, 0, 0, 0, 0, 0],  # party4 is partially missing
    # ], ---epoch 0
    # [
    #  [0, 0, 0, 0, 0, 0, 0],  # party1 is partially missing
    #  [0, 0, 0, 0, 0, 0, 0],  # party2 is partially missing
    #  [0, 0, 0, 0, 0, 0, 0],  # party3 is partially missing
    #  [0, 0, 0, 0, 0, 0, 0],  # party4 is partially missing
    # ], ---epoch 1
    #  ...
    # ]

    # correct_ratio = [[[0] * 7 for _ in range(4)] for _ in range(args.epochs)]
    # correct = [[[0] * 7 for _ in range(4)] for _ in range(args.epochs)]
    # correct = correct
    
    # loss = loss
    
    # train and update
    # ==== Baseline：every client missing_ratio = 0 ====
    fx1, fx2, fx3, fx4 = clients_embeddings[0]
    _, fx_server_all_full = server_model(fx1, fx2, fx3, fx4)
    
    loss_normal = criterion(fx_server_all_full, Y_1).item()
    correct_num = (fx_server_all_full.argmax(1) == Y_1).type(torch.float).sum().item()  
    for party_id in range(len(test_correct[t])):
        test_correct[t][party_id][0] += correct_num
        test_acc[t][party_id][0] = test_correct[t][party_id][0] / size
        
        test_loss[t][party_id][0] += loss_normal
        
    for party_id in range(len(test_correct[t])):  
        for i in range(1, len(test_correct[t][0])):  
            
            fx_parts = []
            for j in range(4):
                if j == party_id:
                    # clients_embeddings 7(or 5)*4
                    fx_parts.append(clients_embeddings[i][j])  # varying
                else:
                    fx_parts.append(clients_embeddings[0][j])  # fixed full input
                    
            _, fx_server_missing = server_model(fx_parts[0], fx_parts[1], fx_parts[2], fx_parts[3])
            
            loss_missing = criterion(fx_server_missing, Y_1).item()
            correct_num = (fx_server_missing.argmax(1) == Y_1).type(torch.float).sum().item()
            
            test_correct[t][party_id][i] += correct_num
            test_acc[t][party_id][i] = test_correct[t][party_id][i] / size
            
            test_loss[t][party_id][i] += loss_missing
            
    if batch_id == len(dataloader) - 1:
        
        for i in range(len(test_loss[t])):
            for j in range(len(test_loss[t][0])):
                test_loss[t][i][j] = test_loss[t][i][j] / len(dataloader)
                
        if test_acc[t][0][0] > best_correct_ratio:
            best_correct_ratio = test_acc[t][0][0]  
            save_path1_best = f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client1_best.pth'
            save_path2_best = f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client2_best.pth'
            save_path3_best = f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client3_best.pth'
            save_path4_best = f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client4_best.pth'
            save_server_best = f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/server_best.pth'
            torch.save(client_model_1.state_dict(), save_path1_best)
            torch.save(client_model_2.state_dict(), save_path2_best)
            torch.save(client_model_3.state_dict(), save_path3_best)
            torch.save(client_model_4.state_dict(), save_path4_best)
            torch.save(server_model.state_dict(), save_server_best)

        if args.dataset =='cifar10':
            missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        elif args.dataset =='MIMIC':
            missing_ratios = [0, 0.3, 0.5, 0.7, 1]
        
        for party_num in range(4):
            
            print(f"---Party{party_num + 1} is Partially Missing---", file=filename)
            for i, missing_ratio in enumerate(missing_ratios):
                
                print(f"Missing ratio: {missing_ratio}", file=filename)
                print(f"Test-loss: {test_loss[t][party_num][i]:>7f}  Test-Accuracy: {(100 * test_acc[t][party_num][i]):>0.1f}%", file=filename)

        print("---Missing Average Case---", file=filename)
        for i, missing_ratio in enumerate(missing_ratios):
            
            total = 0.0
            for row in test_acc[t]:
                total += row[i]
            correct_average_ratio = total / len(test_acc[t])
            
            print(f"---Missing Ratio: {missing_ratio}---", file=filename)
            print(f"Test-Accuracy: {(100 * correct_average_ratio):>0.1f}%", file=filename)
                
def test_client(dataloader, client_model_1, client_model_2, client_model_3, client_model_4, server_model, test_acc, test_loss, test_correct, t):

    client_model_1.eval()
    client_model_2.eval()
    client_model_3.eval()
    client_model_4.eval()
    
    size = len(dataloader) * args.batch_size

    for batch_id, batch in enumerate(dataloader):
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
        
        clients_embeddings = []
        
        half_width = int(X.shape[-1]/2)
        half_height = int(X.shape[-2]/2)
        
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
            
            fx1 = client_model_1(X1_blank)
            fx2 = client_model_2(X2_blank)
            fx3 = client_model_3(X3_blank)
            fx4 = client_model_4(X4_blank)
            
            clients_embeddings.append([
                fx1.clone().detach().requires_grad_(True),
                fx2.clone().detach().requires_grad_(True),
                fx3.clone().detach().requires_grad_(True),
                fx4.clone().detach().requires_grad_(True),
            ])
                           
        # Sending activations to server and receiving gradients from server
        test_server(dataloader, clients_embeddings,server_model, Y_1, batch_id, size, test_acc, test_loss, test_correct, t)
    
# [[
#  [0, 0, 0, 0, 0, 0, 0],  # party1 is partially missing
#  [0, 0, 0, 0, 0, 0, 0],  # party2 is partially missing
#  [0, 0, 0, 0, 0, 0, 0],  # party3 is partially missing
#  [0, 0, 0, 0, 0, 0, 0],  # party4 is partially missing
# ], ---epoch 0
# [
#  [0, 0, 0, 0, 0, 0, 0],  # party1 is partially missing
#  [0, 0, 0, 0, 0, 0, 0],  # party2 is partially missing
#  [0, 0, 0, 0, 0, 0, 0],  # party3 is partially missing
#  [0, 0, 0, 0, 0, 0, 0],  # party4 is partially missing
# ], ---epoch 1
#  ...
# ]
def top_k_mean(acc, k = 5):

    result = [[0.0 for _ in range(len(acc[0][0]))] for _ in range(len(acc[0]))]

    for party in range(len(acc[0])):
        for ratio in range(len(acc[0][0])):
            
            values = [acc[epoch][party][ratio] for epoch in range(len(acc))]
            top_k = sorted(values, reverse=True)[:k]
            result[party][ratio] = sum(top_k) / len(top_k)

    return result
    
   
if __name__ == '__main__':
    # Define record path
    root_path = '.'
    data_path = os.path.join(root_path, 'Dataset').replace('\\', '/')
    save_path = f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = open(f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/A_training_process.txt', 'w+')
    
    if args.dataset == 'utkface' or args.dataset == 'celeba':     
        train_data_nondataloader, test_data, num_classes, num_classes2, channel, hidden = load_data(args.dataset, args.attributes, data_path, args.batch_size)
    elif args.dataset == "imagenet":
        train_data_nondataloader, test_data, num_classes, channel, hidden = get_tinyimagenet_bothloader(batch_size=args.batch_size, shuffle=True, seed=args.seed, device=device, classes = args.classes)
    elif args.dataset =='MIMIC':
        train_data_nondataloader, test_data, input_dim1, input_dim2, input_dim3, input_dim4, num_classes = load_data_tabular_4clients(args.dataset, args.batch_size)
    else: 
        train_data_nondataloader, test_data, num_classes, channel, hidden = gen_dataset(args.dataset, data_path, device = device)

    if args.dataset =='cifar10' or args.dataset =='imagenet' or args.dataset =='utkface':
        ImageModel_init = ImageModel_4clients(dataset= args.dataset, hidden=hidden, num_cutlayer=args.num_cutlayer, num_classes=num_classes, mode='cat', device = device)
        client_model_1, client_model_2, client_model_3, client_model_4, server_model = ImageModel_init.GetModel()
    elif args.dataset =='MIMIC':
        client_model_1, client_model_2, client_model_3, client_model_4, server_model = def_tabular_model_4clients(dataset= args.dataset, level= args.level, input_dim1=input_dim1, input_dim2=input_dim2, input_dim3=input_dim3, input_dim4=input_dim4, num_classes=num_classes, num_cutlayer=args.num_cutlayer, mode='cat', device=device)

    # criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_client1 = torch.optim.Adam(client_model_1.parameters(), lr=args.lr, foreach=False)
    optimizer_client2 = torch.optim.Adam(client_model_2.parameters(), lr=args.lr, foreach=False)
    optimizer_client3 = torch.optim.Adam(client_model_3.parameters(), lr=args.lr, foreach=False)
    optimizer_client4 = torch.optim.Adam(client_model_4.parameters(), lr=args.lr, foreach=False)

    optimizer_server  = torch.optim.Adam(server_model.parameters(),   lr=args.lr, foreach=False)  

    # record results
    train_overlap_loss = []
    train_overlap_acc = []
    
    if args.dataset =='cifar10':
        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    elif args.dataset =='MIMIC':
        missing_ratios = [0, 0.3, 0.5, 0.7, 1]
        
    test_loss = [[[0] * len(missing_ratios) for _ in range(4)] for _ in range(args.epochs)]
    test_acc = [[[0] * len(missing_ratios) for _ in range(4)] for _ in range(args.epochs)]
    test_correct = [[[0] * len(missing_ratios) for _ in range(4)] for _ in range(args.epochs)]

    # start training
    for t in range(args.epochs):
        print(f"Epoch {t + 1}\n-------------------------------", file=filename)

        if args.dataset == 'cifar10' or args.dataset == "imagenet" or args.dataset == 'utkface' or args.dataset =='MIMIC' or args.dataset == 'bank' or args.dataset == 'avazu':
            train_data = get_train(train_data_nondataloader, args.batch_size, args.seed, device) 

        train_client(train_data, client_model_1, client_model_2, client_model_3, client_model_4, server_model, t)
        test_client(test_data, client_model_1, client_model_2, client_model_3, client_model_4, server_model, test_acc, test_loss, test_correct, t)
    
    print("Done!", file=filename)
    
    k = 5
    top_k_mean_results = top_k_mean(test_acc, k = k)
    top_k_mean_average_acc = [0.0] * len(missing_ratios)
    
    print(f"-----Top{k} result------", file=filename)
    
    if args.dataset =='cifar10':
        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    elif args.dataset =='MIMIC':
        missing_ratios = [0, 0.3, 0.5, 0.7, 1]
        
    for i, missing_ratio in enumerate(missing_ratios):
        print(f"Missing_ratio: {missing_ratio}", file=filename)
        
        print(f"top{k}avg--test_party1_missing_acc: {top_k_mean_results[0][i]}", file=filename)
        print(f"top{k}avg--test_party2_missing_acc: {top_k_mean_results[1][i]}", file=filename)
        print(f"top{k}avg--test_party3_missing_acc: {top_k_mean_results[2][i]}", file=filename)
        print(f"top{k}avg--test_party4_missing_acc: {top_k_mean_results[3][i]}", file=filename)
        sum = 0
        for j in range(4):
            sum += top_k_mean_results[j][i]
        
        top_k_mean_average_acc[i] = sum / 4

        print("-------Average-------", file=filename)
        print(f"top{k}avg--test_average_missing_acc: {top_k_mean_average_acc[i]}\n", file=filename)
        
    # save results
    np.save(f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/train_overlap_acc.npy',train_overlap_acc) 
    np.save(f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/train_overlap_loss.npy',train_overlap_loss)
    np.save(f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/test_acc.npy',test_acc) 
    np.save(f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/test_loss.npy',test_loss) 
    np.save(f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/top_k_mean_results.npy',top_k_mean_results)
    np.save(f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/top_k_mean_average_acc.npy',top_k_mean_average_acc)
    
    #plt
    x = np.arange(len(train_overlap_acc))
    plt.plot(x, train_overlap_acc, label='train_overlap_acc')
    plt.xlabel('epoch', fontsize=19)
    plt.ylabel('ACC', fontsize=19)
    plt.title(f'Train_acc_{args.dataset}_sample{args.num_overlap}', fontsize=20)
    plt.legend()
    plt.savefig(f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/Train_Accuracy.png')
    plt.close()

    test_acc_np = np.array(test_acc)  
    test_mean_average_acc = np.mean(test_acc_np, axis=1).T  
    x = np.arange(len(test_acc))  

    save_dir = f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}'
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))

    for i, missing_ratio in enumerate(missing_ratios):
        plt.plot(x, test_mean_average_acc[i], marker='o', linewidth=2, label=f'Missing Ratio {missing_ratio}')

    plt.title('Test Mean Accuracy Over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Test_Mean_Average.png'), dpi=300)
    plt.close()



