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
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
device = torch.device(f"cuda:0") 

def train_server_overlap(client1_fx, Y_1, t, batch_id, correct_overlap, size):
    server_model.train()
    correct = correct_overlap

    _, server_fx_avg = server_model(client1_fx, client1_fx)
    loss_ce = criterion(server_fx_avg, Y_1) 

    loss = loss_ce 

    optimizer_server.zero_grad()
    loss.backward()
    dfx1 = client1_fx.grad.clone().detach().to(device)
    optimizer_server.step()
    
    correct += (server_fx_avg.argmax(1) == Y_1).type(torch.float).sum().item()
     
    if batch_id == args.num_batch-1: 
        save_path3 = f'Result/Results_standalone1/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/server.pth'
        torch.save(server_model, save_path3)
        
            
    if (t+1) %10 ==0 or t == args.epochs-1:
        if batch_id == args.num_batch -1:
            save_path3 = f'Result/Results_standalone1/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/server_epoch{t+1}.pth'
            torch.save(server_model, save_path3)
        
    if batch_id == args.num_batch-1:
        correct_train = correct / (args.num_batch*args.num_overlap)
        loss, current = loss.item(), (batch_id + 1) *args.batch_size
        print(f"train-overlap-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%", file=filename)
        train_overlap_acc.append(100 * correct_train)
        train_overlap_loss.append(loss)

    return dfx1, correct

def train_server_complete2(client1_fx, Y_1, t, batch_id, correct_complete2, size):
    server_model.train()
    correct = correct_complete2

    _, server_fx1 = server_model(client1_fx, client1_fx)

    loss_ce  =  criterion(server_fx1, Y_1)  
    loss = loss_ce 

    optimizer_server.zero_grad()
    loss.backward()
    dfx1 = client1_fx.grad.clone().detach().to(device)
    optimizer_server.step()

    correct += (server_fx1.argmax(1) == Y_1).type(torch.float).sum().item()
    
    if batch_id == args.num_batch-1: 
        save_path3 = f'Result/Results_standalone1/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/server.pth'
        torch.save(server_model, save_path3)
    
    if (t+1) %10 ==0 or t == args.epochs-1:
        if batch_id == args.num_batch-1:
            save_path3 = f'Result/Results_standalone1/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/server_epoch{t+1}.pth'
            torch.save(server_model, save_path3)
            
    if batch_id == args.num_batch-1:
        correct_train = correct / ((args.num_batch*args.batch_size - args.num_batch*args.num_overlap) * (1 - args.nonoverlap1_ratio))
        loss, current = loss.item(), (batch_id + 1) *args.batch_size
        print(f"train-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%",
            file=filename)
        train_complete2_acc.append(100 * correct_train)
        train_complete2_loss.append(loss)

    return dfx1, correct

def train_client(dataloader, client_model_1, client_model_2, t):
    client_model_1.train()
    client_model_2.train()

    correct_overlap= 0
    correct_complete2= 0

    size = len(dataloader)*args.batch_size

    for batch_id, batch in enumerate(dataloader):
      
        X, target = batch
        
        X_1, X_2 = split_data(args.dataset, X)
        
        if args.dataset == 'utkface' or args.dataset == 'celeba':
            Y_1 = target[0].to(device)

            Y_2 = target[1].view(-1, 1).to(device)
        else:
            Y_1 = target.to(device)
        
        # x_overlap & non-overlap
        X_1_overlap     = X_1[:args.num_overlap]
        X_2_overlap     = X_2[:args.num_overlap]
        X_1_non_overlap = X_1[int((args.batch_size - args.num_overlap)*args.nonoverlap1_ratio + args.num_overlap)  : args.batch_size]
        X_2_non_overlap = X_2[args.num_overlap:int((args.batch_size - args.num_overlap)*args.nonoverlap1_ratio + args.num_overlap)]

        
        Y_overlap = Y_1[:args.num_overlap]
        Y_non_overlap2 = Y_1[args.num_overlap:int((args.batch_size - args.num_overlap)*args.nonoverlap1_ratio + args.num_overlap)]
        Y_non_overlap1 = Y_1[int((args.batch_size - args.num_overlap)*args.nonoverlap1_ratio + args.num_overlap):args.batch_size]

        if args.overlap == 'True':

            fx1 = client_model_1(X_1_overlap)

            client1_fx = (fx1).clone().detach().requires_grad_(True)

            g_fx1, correct_overlap = train_server_overlap(client1_fx,  Y_overlap, t, batch_id, correct_overlap, size)
            
            optimizer_client1.zero_grad()
            (fx1).backward(g_fx1)
            optimizer_client1.step()

        if args.complete_2 == 'True':
           
            fx1 = client_model_1(X_1_non_overlap)

            client1_fx   = (fx1).clone().detach().requires_grad_(True)
            g_fx1, correct_complete2 = train_server_complete2(client1_fx, Y_non_overlap1, t, batch_id, correct_complete2, size)
            
            optimizer_client1.zero_grad()
            (fx1).backward(g_fx1, retain_graph=True)
            optimizer_client1.step()

        elif args.complete_2 == 'False' and batch_id == args.num_batch-1:
            train_complete2_acc.append(0)
            train_complete2_loss.append(0)

        if (t+1) %10 ==0 or t == args.epochs-1:
            if batch_id == args.num_overlap-1:  
                save_path1 = f'Result/Results_standalone1/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client1_epoch{t+1}.pth'
                save_path2 = f'Result/Results_standalone1/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client2_epoch{t+1}.pth'
                torch.save(client_model_1, save_path1)
                torch.save(client_model_2, save_path2)
            
        if batch_id >= args.num_batch -1:
            break
        
def test_server(client1_fx, partyA_zeros, Y_1, 
                batch_id, correct_normal, correct_zeros, size):
    server_model.eval()
    correct_normal = correct_normal
    correct_zeros = correct_zeros

    optimizer_server.zero_grad()
    _, fx_server_normal = server_model(client1_fx, client1_fx)
    loss_normal = criterion(fx_server_normal, Y_1).item()
    correct_normal += (fx_server_normal.argmax(1) == Y_1).type(torch.float).sum().item()
     
    loss_zeros = [0] * len(partyA_zeros) 
    for i, partyA_zero in enumerate(partyA_zeros):
        optimizer_server.zero_grad()
        _, fx_server_zero = server_model(partyA_zero, partyA_zero)
        loss_zeros[i] = criterion(fx_server_zero, Y_1).item()
        correct_zeros[i] += (fx_server_zero.argmax(1) == Y_1).type(torch.float).sum().item()

    correct_normal_ratio = (correct_normal / size)
    current = (batch_id + 1) * len(Y_1)

    if batch_id == len(test_data) - 1:
        print("---Normal Case---",file=filename)
        print(f"Test-loss: {loss_normal:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_normal_ratio):>0.1f}%",
              file=filename)
        
        test_normal_acc.append(100 * correct_normal_ratio)
        test_normal_loss.append(loss_normal)

        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        print("---Zero Case---",file=filename)
        for i, missing_ratio in enumerate(missing_ratios):
            correct_zero_ratio = (correct_zeros[i] / size)
            print(f"Missing ratio: {missing_ratio}", file=filename)

            print(f"Test-loss: {loss_zeros[i]:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_zero_ratio):>0.1f}%",
                  file=filename)
            test_zero_acc_cases[i].append(100 * correct_zero_ratio)
            test_zero_loss_cases[i].append(loss_zeros[i])

    return correct_normal, correct_zeros

def test_client(dataloader, client_model_1, client_model_2, t):
    client_model_1.eval()
    client_model_2.eval()

    correct_normal = 0
    correct_zeros = [0] * 7

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

        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        partyA_zeros = [] 
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

            fx1_blank = client_model_1(X_1_blank)
            partyA_zero = (fx1_blank).clone().detach().requires_grad_(True)
            partyA_zeros.append(partyA_zero)
        # Sending activations to server and receiving gradients from server
        correct_normal, correct_zeros = test_server(client1_fx, partyA_zeros, Y_1, 
                                                   batch_id, correct_normal, correct_zeros, size)

if __name__ == '__main__':
    # Define record path
    root_path = '.'
    data_path = os.path.join(root_path, 'Dataset').replace('\\', '/')
    save_path = f'Result/Results_standalone1/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = open(f'Result/Results_standalone1/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/A_training_process.txt', 'w+')

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
      

    if args.dataset =='MIMIC' or args.dataset == 'avazu' or args.dataset == 'bank':
        client_model_1, client_model_2, server_model = def_tabular_model(dataset= args.dataset, level= args.level, input_dim1=input_dim1, input_dim2=input_dim2, num_classes=num_classes, num_cutlayer=args.num_cutlayer, mode='standalone')
    else: 
        ImageModel1= ImageModel(dataset= args.dataset, hidden=hidden, num_cutlayer=args.num_cutlayer, num_classes=num_classes, mode='standalone', device=device)
        client_model_1, client_model_2, server_model = ImageModel1.GetModel()
        

    # criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_client1 = torch.optim.Adam(client_model_1.parameters(), lr=args.lr, foreach=False )
    optimizer_client2 = torch.optim.Adam(client_model_2.parameters(), lr=args.lr, foreach=False )
    optimizer_server  = torch.optim.Adam(server_model.parameters(),   lr=args.lr, foreach=False )  

    # record results
    train_overlap_loss = []
    train_overlap_acc = []

    train_complete2_loss = []
    train_complete2_acc = []

    test_normal_loss = []
    test_normal_acc = []
    test_zero_acc_cases = [[] for _ in range(7)] 
    test_zero_loss_cases = [[] for _ in range(7)] 

    # start training
    for t in range(args.epochs):
        print(f"Epoch {t + 1}\n-------------------------------", file=filename)
        print(t)

        if args.dataset == 'cifar10' or args.dataset == "imagenet" or args.dataset == 'utkface' or args.dataset =='MIMIC' or args.dataset == 'bank' or args.dataset == 'avazu':
            train_data = get_train(train_data_nondataloader, args.batch_size, args.seed, device) 

        train_client(train_data, client_model_1, client_model_2, t)
        test_client(test_data, client_model_1, client_model_2, t)
    print("Done!", file=filename)

    print("-----Max result------", file=filename)
    print("max--test_normal_acc!", np.array(test_normal_acc).max(), file=filename)
    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        print(f"Missing_ratio: {missing_ratio}", file=filename)
        print("max--test_zero_acc!", np.array(test_zero_acc_cases[i]).max(), file=filename)

    k = 5
    max_k_test_zero_acc = [[] for _ in range(7)]

    max_k_test_normal_acc = np.mean(np.sort(test_normal_acc)[-k:])
    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        max_k_test_zero_acc[i] = np.mean(np.sort(test_zero_acc_cases[i])[-k:])

    print(f"-----Top{k} result------", file=filename)
    print(f"top{k}avg--test_normal_acc: {max_k_test_normal_acc}", file=filename)

    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        print(f"Missing_ratio: {missing_ratio}", file=filename)
        print(f"top{k}avg--test_average_acc: {max_k_test_zero_acc[i]}", file=filename)
    
    #save
    np.save(f'Result/Results_standalone1/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/train_overlap_ACC_sum.npy',   train_overlap_acc) 
    np.save(f'Result/Results_standalone1/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/train_overlap_lOSS_sum.npy',  train_overlap_loss)
    np.save(f'Result/Results_standalone1/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/train_complete2_ACC_sum.npy',   train_complete2_acc) 
    np.save(f'Result/Results_standalone1/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/train_complete2_lOSS_sum.npy',  train_complete2_loss)
     
    np.save(f'Result/Results_standalone1/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/test_normal_acc.npy',   test_normal_acc) 
    np.save(f'Result/Results_standalone1/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/test_normal_loss.npy',  test_normal_loss) 
    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        np.save(f'Result/Results_standalone1/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/test_zero_acc_Missing{missing_ratio}.npy',   test_zero_acc_cases[i]) 
        np.save(f'Result/Results_standalone1/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/test_zero_loss_Missing{missing_ratio}.npy',  test_zero_loss_cases[i]) 
    
    #plt
    x = np.arange(len(train_overlap_acc))
    plt.plot(x, train_overlap_acc,   label='train_overlap_acc')
    plt.plot(x, train_complete2_acc, label='train_complete2_acc')
    plt.plot(x, test_normal_acc,  label='test_normal_acc')
    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        plt.plot(x, test_zero_acc_cases[i],  label=f'test_zero_acc_Missing{missing_ratio}')
    plt.xlabel('epoch',   fontsize=19)
    plt.ylabel('ACC',   fontsize=19)
    plt.title(f'{args.dataset}_sample{args.num_overlap}',   fontsize=20)
    plt.legend()
    plt.savefig(f'Result/Results_standalone1/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/A_train_ACC.png')
    plt.close()
   
    x = np.arange(len(train_overlap_loss))
    plt.plot(x, train_overlap_loss, label='train_overlap_loss')
    plt.plot(x, train_complete2_loss, label='train_complete2_loss')
    plt.plot(x, test_normal_loss, label='test_normal_loss')
    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        plt.plot(x, test_zero_loss_cases[i], label=f'test_zero_loss_Missing{missing_ratio}')
    plt.xlabel('epoch',   fontsize=19)
    plt.ylabel('Loss',   fontsize=19)
    plt.title(f'{args.dataset}_sample{args.num_overlap}',   fontsize=20)
    plt.legend()
    plt.savefig(f'Result/Results_standalone1/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/A_train_Loss.png')
    plt.close()
    
    

    
    
    
    
    





