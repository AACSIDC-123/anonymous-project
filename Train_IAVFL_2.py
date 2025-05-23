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

# Define A random_seed
set_random_seed(args.seed)
best_correct_ratio = 0.0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
device = torch.device(f"cuda:0") 

# Train_Server Side Program
def train_server_overlap(client2_student_fx, client1_teacher_fx, client2_teacher_fx, Y_1, t, batch_id, correct_overlap, size, server_teacher_model, server_student_model):
    
    server_student_model.train()
    server_teacher_model.eval()

    correct = correct_overlap
    
    T = 2.0  
    _, server_student = server_student_model(client2_student_fx, client2_student_fx)
    loss_student_ce = criterion(server_student, Y_1) 

    _, server_teacher = server_teacher_model(client1_teacher_fx, client2_teacher_fx)
    soft_labels_teacher = F.softmax(server_teacher / T, dim=1) 

    loss_teacher_ce = F.kl_div(
        F.log_softmax(server_student / T, dim=1), 
        soft_labels_teacher,                          
    ) * (T * T)  

    alpha = 0.5  
    loss = (1 - alpha) * loss_student_ce + alpha * loss_teacher_ce
    
    # backward
    optimizer_student_server.zero_grad()
    loss.backward()
    dfx2 = client2_student_fx.grad.clone().detach().to(device)
    optimizer_student_server.step()
    
    #acc
    correct += (server_student.argmax(1) == Y_1).type(torch.float).sum().item()
     
    if batch_id == args.num_batch-1:
        correct_train = correct / (args.num_batch*args.num_overlap)
        loss, current = loss.item(), (batch_id + 1) *args.batch_size
        print(f"train-overlap-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%", file=filename)
        train_overlap_acc.append(100 * correct_train)
        train_overlap_loss.append(loss)

    return dfx2, correct

# Train_Client Side Program
def train_client(dataloader, client_student_model_1, client_student_model_2, server_student_model, \
            client_teacher_model_1, client_teacher_model_2, server_teacher_model, t):
    
    # load the trained cat model
    save_path1_best = f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{int(args.num_cutlayer/2)}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client1_best.pth'
    save_path2_best = f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{int(args.num_cutlayer/2)}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client2_best.pth'
    save_path3_best = f'Result/Results_cat/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{int(args.num_cutlayer/2)}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/server_best.pth'
    
    server_teacher_model.load_state_dict(torch.load(save_path3_best, map_location=device))
    client_teacher_model_1.load_state_dict(torch.load(save_path1_best,map_location=device))
    client_teacher_model_2.load_state_dict(torch.load(save_path2_best,map_location=device))

    client_student_model_1.train()
    client_student_model_2.train()
    server_student_model.train()

    client_teacher_model_1.eval()
    client_teacher_model_2.eval()
    server_teacher_model.eval()

    correct_overlap = 0
    correct_complete1 = 0

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
        
        X_2_non_overlap = X_2[args.num_overlap:int((args.batch_size - args.num_overlap)*args.nonoverlap1_ratio + args.num_overlap)]
        X_1_non_overlap = X_1[int((args.batch_size - args.num_overlap)*(args.nonoverlap1_ratio) + args.num_overlap):args.batch_size]

        Y_overlap = Y_1[:args.num_overlap]
        Y_non_overlap2 = Y_1[args.num_overlap:int((args.batch_size - args.num_overlap)*args.nonoverlap1_ratio + args.num_overlap)]
        Y_non_overlap1 = Y_1[int((args.batch_size - args.num_overlap)*(args.nonoverlap1_ratio) + args.num_overlap):args.batch_size]
        
        # client1--train and update
        if args.overlap == 'True':
            #E
            fx2_student = client_student_model_2(X_2_overlap)
            fx1_teacher = client_teacher_model_1(X_1_overlap)
            fx2_teacher = client_teacher_model_2(X_2_overlap)
            # Sending activations to server and receiving gradients from server
            client2_student_fx = (fx2_student).clone().detach().requires_grad_(True)
            client1_teacher_fx = (fx1_teacher).clone().detach().requires_grad_(True)
            client2_teacher_fx = (fx2_teacher).clone().detach().requires_grad_(True)

            g_fx2, correct_overlap = train_server_overlap(client2_student_fx, client1_teacher_fx, client2_teacher_fx, Y_overlap, t, batch_id, correct_overlap, size, server_teacher_model, server_student_model)
            
            #backward
            optimizer_student_client2.zero_grad()
            (fx2_student).backward(g_fx2)
            optimizer_student_client2.step()
        
        if args.complete_1 == 'True':
            #E
            fx2 = client_student_model_2(X_2_non_overlap)

            # Sending activations to server and receiving gradients from server
            client2_fx = (fx2).clone().detach().requires_grad_(True)
            g_fx2, correct_complete1 = train_server_complete1(client2_fx, Y_non_overlap2, t, batch_id, correct_complete1, server_student_model, size)
            
            optimizer_student_client2.zero_grad()
            (fx2).backward(g_fx2, retain_graph=True)
            optimizer_student_client2.step()
        
        elif args.complete_1 == 'False' and batch_id == args.num_batch-1:
            train_complete1_acc.append(0)
            train_complete1_loss.append(0)    
        
        # record for attack
        if batch_id == args.num_batch -1:
            save_path1 = f'Result/Results_IAVFL_2/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client1.pth'
            save_path2 = f'Result/Results_IAVFL_2/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client2.pth'
            save_path3 = f'Result/Results_IAVFL_2/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/server.pth'
            
            torch.save(client_student_model_1.state_dict(), save_path1)
            torch.save(client_student_model_2.state_dict(), save_path2)
            torch.save(server_student_model.state_dict(), save_path3)
          
        if batch_id >= args.num_batch-1:
            break
        
def train_server_complete1(client2_fx, Y_1, t, batch_id, correct_complete1, server_student_model, size):
    
    server_student_model.train()
    correct = correct_complete1
    
    # train and update
    _, server_fx2 = server_student_model(client2_fx, client2_fx)

    loss_ce  =  criterion(server_fx2, Y_1)  
    loss = loss_ce 
    
    # backward
    optimizer_student_server.zero_grad()
    loss.backward()
    dfx2 = client2_fx.grad.clone().detach().to(device)
    optimizer_student_server.step()
    
    #acc
    correct += (server_fx2.argmax(1) == Y_1).type(torch.float).sum().item()
    
    if batch_id == args.num_batch-1:
        correct_train = correct / ((args.num_batch*args.batch_size - args.num_batch*args.num_overlap) * args.nonoverlap1_ratio)
        loss, current = loss.item(), (batch_id + 1) *args.batch_size
        print(f"train-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%",
            file=filename)
        train_complete1_acc.append(100 * correct_train)
        train_complete1_loss.append(loss)

    return dfx2, correct

def test_server(client2_fx, partyB_zeros, Y_1, 
                batch_id, correct_normal, correct_zeros, server_student_model, size):
    
    server_student_model.eval()
    correct_normal = correct_normal
    correct_zeros = correct_zeros

    global best_correct_ratio
    
    # train and update
    optimizer_student_server.zero_grad()
    _, fx_server_normal = server_student_model(client2_fx, client2_fx)
    loss_normal = criterion(fx_server_normal, Y_1).item()
    correct_normal += (fx_server_normal.argmax(1) == Y_1).type(torch.float).sum().item()
    
    loss_zeros = [0] * len(partyB_zeros) 
    for i, partyB_zero in enumerate(partyB_zeros):
        optimizer_student_server.zero_grad()
        _, fx_server_zero = server_student_model(partyB_zero, partyB_zero)
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
        
        if correct_normal_ratio > best_correct_ratio:
            print("best model update\n", file = filename)
            best_correct_ratio = correct_normal_ratio  
            save_path1_best = f'Result/Results_IAVFL_2/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client1_best.pth'
            save_path2_best = f'Result/Results_IAVFL_2/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client2_best.pth'
            save_path3_best = f'Result/Results_IAVFL_2/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/server_best.pth'
            torch.save(client_student_model_1.state_dict(), save_path1_best)
            torch.save(client_student_model_2.state_dict(), save_path2_best)
            torch.save(server_student_model.state_dict(), save_path3_best)

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

# Test_Client Side Program
def test_client(dataloader, client_student_model_1, client_student_model_2, server_student_model, t):
    
    client_student_model_1.eval()
    client_student_model_2.eval()
    server_student_model.eval()
    
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

        fx2 = client_student_model_2(X_2)
        client2_fx = (fx2).clone().detach().requires_grad_(True)
        
        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
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

            fx2_blank = client_student_model_2(X_2_blank)
            partyB_zero = (fx2_blank).clone().detach().requires_grad_(True)
            partyB_zeros.append(partyB_zero)
        # Sending activations to server and receiving gradients from server
        correct_normal, correct_zeros = test_server(client2_fx, partyB_zeros, Y_1, 
                                                   batch_id, correct_normal, correct_zeros, server_student_model, size)

if __name__ == '__main__':
    # Define record path
    root_path = '.'
    data_path = os.path.join(root_path, 'Dataset').replace('\\', '/')
    save_path = f'Result/Results_IAVFL_2/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = open(f'Result/Results_IAVFL_2/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/A_training_process.txt', 'w+')
    
    ### Load data 
    if args.dataset == 'utkface' or args.dataset == 'celeba':      
        train_data_nondataloader, test_data, num_classes, num_classes2, channel, hidden = load_data(args.dataset, args.attributes, data_path, args.batch_size)
        
    elif args.dataset =='MIMIC' or args.dataset == 'avazu' or args.dataset == 'bank':
        train_data_nondataloader, test_data, input_dim1, input_dim2, num_classes = load_data_tabular(args.dataset, args.batch_size)

    elif args.dataset == "imagenet":
        train_data_nondataloader, test_data, num_classes, channel, hidden = get_tinyimagenet_bothloader(batch_size=args.batch_size, shuffle=True, seed=args.seed, device=device, classes = args.classes)

    else:
        train_data_nondataloader, test_data, num_classes, channel, hidden = gen_dataset(args.dataset, data_path, device = device)

    
    if args.dataset =='MIMIC' or args.dataset == 'avazu' or args.dataset == 'bank':
        client_student_model_1, client_student_model_2, server_student_model = def_tabular_model(dataset= args.dataset, level= args.level, input_dim1=input_dim1, input_dim2=input_dim2, num_classes=num_classes, num_cutlayer=args.num_cutlayer, mode='standalone')
        client_teacher_model_1, client_teacher_model_2, server_teacher_model = def_tabular_model(dataset= args.dataset, level= args.level, input_dim1=input_dim1, input_dim2=input_dim2, num_classes=num_classes, num_cutlayer=int(args.num_cutlayer/2), mode='cat')
    
    else:
        ImageModel1= ImageModel(dataset= args.dataset, hidden=hidden, num_cutlayer=args.num_cutlayer, num_classes=num_classes, mode='standalone', device=device)
        client_student_model_1, client_student_model_2, server_student_model = ImageModel1.GetModel()

        ImageModel2= ImageModel(dataset= args.dataset, hidden=hidden, num_cutlayer=int(args.num_cutlayer/2), num_classes=num_classes, mode='cat', device=device)
        client_teacher_model_1, client_teacher_model_2, server_teacher_model = ImageModel2.GetModel()
        
    # criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_student_client1 = torch.optim.Adam(client_student_model_1.parameters(), lr=args.lr, foreach=False )
    optimizer_student_client2 = torch.optim.Adam(client_student_model_2.parameters(), lr=args.lr, foreach=False )
    optimizer_student_server  = torch.optim.Adam(server_student_model.parameters(),   lr=args.lr, foreach=False )  

    # record results
    train_overlap_loss = []
    train_overlap_acc = []
    train_complete1_loss = []
    train_complete1_acc = []

    test_normal_loss = []
    test_normal_acc = []
    test_zero_loss_cases = [[] for _ in range(7)]
    test_zero_acc_cases = [[] for _ in range(7)]

    # start training
    for t in range(args.epochs):
        print(f"Epoch {t + 1}\n-------------------------------", file=filename)
        print(t)

        if args.dataset == 'cifar10' or args.dataset == "imagenet" or args.dataset == 'utkface' or args.dataset =='MIMIC' or args.dataset == 'bank' or args.dataset == 'avazu':
            train_data = get_train(train_data_nondataloader, args.batch_size, args.seed, device) 
      
        train_client(train_data, client_student_model_1, client_student_model_2, server_student_model, \
            client_teacher_model_1, client_teacher_model_2, server_teacher_model, t)
        test_client(test_data, client_student_model_1, client_student_model_2, server_student_model, t)
    print("Done!", file=filename)

    #test_acc
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
    np.save(f'Result/Results_IAVFL_2/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/train_overlap_ACC.npy',   train_overlap_acc) 
    np.save(f'Result/Results_IAVFL_2/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/train_overlap_lOSS.npy',  train_overlap_loss)
    np.save(f'Result/Results_IAVFL_2/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/train_complete1_ACC.npy',   train_complete1_acc) 
    np.save(f'Result/Results_IAVFL_2/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/train_complete1_lOSS.npy',  train_complete1_loss)
     
    np.save(f'Result/Results_IAVFL_2/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/test_normal_acc.npy',   test_normal_acc) 
    np.save(f'Result/Results_IAVFL_2/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/test_normal_loss.npy',  test_normal_loss) 
    
    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        np.save(f'Result/Results_IAVFL_2/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/test_zero_acc_Missing{missing_ratio}.npy',   test_zero_acc_cases[i]) 
        np.save(f'Result/Results_IAVFL_2/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/test_zero_loss_Missing{missing_ratio}.npy',  test_zero_loss_cases[i]) 
    
    
    #plt
    x1 = np.arange(len(train_overlap_acc))
    x2 = np.arange(len(train_complete1_acc))
    x3 = np.arange(len(test_normal_acc))
    
    plt.plot(x1, train_overlap_acc, label='train_overlap_acc')
    plt.plot(x2, train_complete1_acc, label='train_complete1_acc')
    plt.plot(x3, test_normal_acc,  label='test_normal_acc')
    
    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        x = np.arange(len(test_zero_acc_cases[i]))
        plt.plot(x, test_zero_acc_cases[i],  label=f'test_zero_acc_Missing{missing_ratio}')
    plt.xlabel('epoch',   fontsize=19)
    plt.ylabel('ACC',   fontsize=19)
    plt.title(f'{args.dataset}_sample{args.num_overlap}',   fontsize=20)
    plt.legend()
    plt.savefig(f'Result/Results_IAVFL_2/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/A_train_ACC.png')
    plt.close()
    
    
    x = np.arange(len(train_overlap_loss))
    plt.plot(x, train_overlap_loss, label='train_overlap_loss')
    plt.plot(x, train_complete1_loss, label='train_complete1_loss')
    plt.plot(x, test_normal_loss, label='test_normal_loss')
    missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for i, missing_ratio in enumerate(missing_ratios):
        plt.plot(x, test_zero_loss_cases[i], label=f'test_zero_loss_Missing{missing_ratio}')
    
    plt.xlabel('epoch',   fontsize=19)
    plt.ylabel('Loss',   fontsize=19)
    plt.title(f'{args.dataset}_sample{args.num_overlap}',   fontsize=20)
    plt.legend()
    plt.savefig(f'Result/Results_IAVFL_2/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/A_train_Loss.png')
    plt.close()
    
    

    
    
    
    
    





