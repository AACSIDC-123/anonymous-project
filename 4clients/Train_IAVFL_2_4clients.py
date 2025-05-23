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

# Define A random_seed
set_random_seed(args.seed)
correct_best = 0
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
    
# Train_Client Side Program
def train_client(dataloader, client_student_model_1, client_student_model_2, \
            client_student_model_3, client_student_model_4, server_student_model, \
            client_teacher_model_1, client_teacher_model_2, client_teacher_model_3, \
            client_teacher_model_4, server_teacher_model, t):
    
    # load the cat trained model
    save_path1_best = f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{int(args.num_cutlayer/4)}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client1_best.pth'
    save_path2_best = f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{int(args.num_cutlayer/4)}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client2_best.pth'
    save_path3_best = f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{int(args.num_cutlayer/4)}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client3_best.pth'
    save_path4_best = f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{int(args.num_cutlayer/4)}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client4_best.pth'
    save_server_best = f'Result/Results_cat_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{int(args.num_cutlayer/4)}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/server_best.pth'
    
    client_teacher_model_1.load_state_dict(torch.load(save_path1_best, map_location=device))
    client_teacher_model_2.load_state_dict(torch.load(save_path2_best, map_location=device))
    client_teacher_model_3.load_state_dict(torch.load(save_path3_best, map_location=device))
    client_teacher_model_4.load_state_dict(torch.load(save_path4_best, map_location=device))
    server_teacher_model.load_state_dict(torch.load(save_server_best, map_location=device))

    client_student_model_1.train()
    client_student_model_2.train()
    client_student_model_3.train()
    client_student_model_4.train()
    server_student_model.train()
    
    client_teacher_model_1.eval()
    client_teacher_model_2.eval()
    client_teacher_model_3.eval()
    client_teacher_model_4.eval()
    server_teacher_model.eval()

    correct_overlap = 0
    correct_non_overlap = 0
    
    loss_overlap = 0
    loss_non_overlap = 0

    size = len(dataloader)*args.batch_size

    for batch_id, batch in enumerate(dataloader):

        X, target = batch
        X_1, X_2, X_3, X_4 = split_data_4clients(args.dataset, X)
        
        if args.dataset == 'utkface' or args.dataset == 'celeba':
            Y_1 = target[0].to(device)
            Y_2 = target[1].view(-1, 1).to(device)
        else:
            Y_1 = target.to(device)
        
        # x_overlap & non-overlap
        X_1_overlap   = X_1[:args.num_overlap]
        X_2_overlap   = X_2[:args.num_overlap]
        X_3_overlap   = X_3[:args.num_overlap]
        X_4_overlap   = X_4[:args.num_overlap]
        
        X_2_complete1 = X_2[args.num_overlap : int(((args.batch_size - args.num_overlap)/4)*1 + args.num_overlap)]
        X_2_complete2 = X_2[int(((args.batch_size - args.num_overlap)/4)*1 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*2 + args.num_overlap)]
        X_2_complete3 = X_2[int(((args.batch_size - args.num_overlap)/4)*2 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*3 + args.num_overlap)]
        X_2_complete4 = X_2[int(((args.batch_size - args.num_overlap)/4)*3 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*4 + args.num_overlap)]

        X_2_non_overlap = torch.cat([X_2_complete1, X_2_complete3, X_2_complete4], dim=0)
        
        Y_overlap = Y_1[:args.num_overlap]
        Y_complete1 = Y_1[args.num_overlap : int(((args.batch_size - args.num_overlap)/4)*1 + args.num_overlap)]
        Y_complete2 = Y_1[int(((args.batch_size - args.num_overlap)/4)*1 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*2 + args.num_overlap)]
        Y_complete3 = Y_1[int(((args.batch_size - args.num_overlap)/4)*2 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*3 + args.num_overlap)]
        Y_complete4 = Y_1[int(((args.batch_size - args.num_overlap)/4)*3 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*4 + args.num_overlap)]
        
        Y_non_overlap = torch.cat([Y_complete1, Y_complete3, Y_complete4], dim=0)
        
        num_complete1 = int((args.batch_size - args.num_overlap)/4)
        num_complete2 = int((args.batch_size - args.num_overlap)/4)
        num_complete3 = int((args.batch_size - args.num_overlap)/4)
        num_complete4 = int((args.batch_size - args.num_overlap)/4)
        
        # -----client2--overlap-----
        # ***overlap_client***
        fx2_student = client_student_model_2(X_2_overlap)
        
        fx1_teacher = client_teacher_model_1(X_1_overlap)
        fx2_teacher = client_teacher_model_2(X_2_overlap)
        fx3_teacher = client_teacher_model_3(X_3_overlap)
        fx4_teacher = client_teacher_model_4(X_4_overlap)
        
        # ***overlap_server***
        T = 2.0  
        _, server_student = server_student_model(fx2_student)
        loss_student_ce = criterion(server_student, Y_overlap) 

        _, server_teacher = server_teacher_model(fx1_teacher, fx2_teacher, fx3_teacher, fx4_teacher)
        soft_labels_teacher = F.softmax(server_teacher / T, dim=1)  

        loss_teacher_ce = F.kl_div(
            F.log_softmax(server_student / T, dim=1), 
            soft_labels_teacher                   
        ) * (T * T)  

        alpha = 0.5  
        loss = (1 - alpha) * loss_student_ce + alpha * loss_teacher_ce

        # backward
        optimizer_student_server.zero_grad()
        optimizer_student_client2.zero_grad()
        loss.backward()
        optimizer_student_server.step()
        optimizer_student_client2.step()
        
        correct_overlap += (server_student.argmax(1) == Y_overlap).type(torch.float).sum().item()
        loss_overlap += loss.item()
        
        # -----client2--non_overlap-----
        # ***non-overlap_client***
        fx2 = client_student_model_2(X_2_non_overlap)

        # ***non-overlap_server***
        _, server_fx2 = server_student_model(fx2)
        
        loss = criterion(server_fx2, Y_non_overlap)  
        
        # backward
        optimizer_student_server.zero_grad()
        optimizer_student_client2.zero_grad()
        loss.backward()
        optimizer_student_server.step()
        optimizer_student_client2.step()
        
        correct_non_overlap += (server_fx2.argmax(1) == Y_non_overlap).type(torch.float).sum().item()
        loss_non_overlap += loss.item() 
        
        # ----------------------record all needed model para and acc&loss------------------
        if batch_id == args.num_batch -1:
            correct_all = correct_overlap + correct_non_overlap
            
            accuracy = correct_all / (args.num_batch*(args.num_overlap+num_complete1+num_complete3+num_complete4))
            loss_all = loss_overlap + loss_non_overlap
            current = (batch_id + 1) *args.batch_size
            
            print(f"Train-loss: {loss_all:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * accuracy):>0.1f}%", file=filename)
            train_acc.append(100 * accuracy)
            train_loss.append(loss_all)
            
        if batch_id >= args.num_batch -1:
            break
        
# Test_Client Side Program
def test_client(dataloader, client_student_model_1, client_student_model_2, \
            client_student_model_3, client_student_model_4, server_student_model, t):
    client_student_model_1.eval()
    client_student_model_2.eval()
    client_student_model_3.eval()
    client_student_model_4.eval()
    server_student_model.eval()
    
    global correct_best
    
    if args.dataset =='cifar10':
        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    elif args.dataset =='MIMIC':
        missing_ratios = [0, 0.3, 0.5, 0.7, 1]
    
    correct = [0] * len(missing_ratios)
    loss = [0] * len(missing_ratios)

    size = len(dataloader) * args.batch_size

    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X_1, X_2, X_3, X_4 = split_data_4clients(args.dataset, X)

        if args.dataset == 'utkface' or args.dataset == 'celeba':
            Y_1 = target[0].to(device)
        else:
            Y_1 = target.to(device)

        if args.dataset =='cifar10':
            missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        elif args.dataset =='MIMIC':
            missing_ratios = [0, 0.3, 0.5, 0.7, 1]
        
        client_2_embeddings = [] 
        
        half_width = int(X.shape[-1]/2)
        half_height = int(X.shape[-2]/2)
        
        for i, missing_ratio in enumerate(missing_ratios):
            
            if missing_ratio > 0:
                if args.dataset == 'cifar10':
                    # Client 2
                    X2_blank = X_2.clone()
                    X2_blank[:, :, int(half_height * (1 - missing_ratio)):, :int(half_width * (1 + missing_ratio))] = 0
                elif args.dataset == 'MIMIC':
                    Missing_item_1, Missing_item_2, Missing_item_3, Missing_item_4 = get_Missing_item_MIMIC(missing_ratio)
                    X2_blank = X_2.clone()
                    X2_blank[:,-Missing_item_2:] = 0
                    
            else:
                X2_blank = X_2

            fx2 = client_student_model_2(X2_blank)
            client_2_embeddings.append(fx2.clone().detach().requires_grad_(True))
        
        for i, client_2_embedding in enumerate(client_2_embeddings):
        
            _, fx_server = server_student_model(client_2_embedding)
            loss[i] += criterion(fx_server, Y_1).item()
            correct[i] += (fx_server.argmax(1) == Y_1).type(torch.float).sum().item()

        current = (batch_id + 1) * len(Y_1)
        if batch_id == len(dataloader) - 1:
            
            if correct[0] > correct_best:
                
                print("best_model updating: -------",file = filename)
                correct_best = correct[0]
                # save_path1 = f'Result/Results_IAVFL_2_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client1_best.pth'
                save_path2 = f'Result/Results_IAVFL_2_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client2_best.pth'
                # save_path3 = f'Result/Results_IAVFL_2_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client3_best.pth'
                # save_path4 = f'Result/Results_IAVFL_2_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client4_best.pth'
                save_server_path = f'Result/Results_IAVFL_2_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/server_best.pth'
                # torch.save(client_student_model_1.state_dict(), save_path1)
                torch.save(client_student_model_2.state_dict(), save_path2)
                # torch.save(client_student_model_3.state_dict(), save_path3)
                # torch.save(client_student_model_4.state_dict(), save_path4)
                torch.save(server_student_model.state_dict(), save_server_path)
            
            
            # calculate acc
            acc = [c / size for c in correct]
            loss_avg = [l / len(dataloader) for l in loss]
            
            for i in range(len(test_acc[0])):
                test_acc[t][i] = acc[i]
                test_loss[t][i] = loss_avg[i]
                
            if args.dataset =='cifar10':
                missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
            elif args.dataset =='MIMIC':
                missing_ratios = [0, 0.3, 0.5, 0.7, 1]

            for i, missing_ratio in enumerate(missing_ratios):
                print(f"Missing ratio is:{missing_ratio}", file=filename)
                print(f"Test-loss: {test_loss[t][i]:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * test_acc[t][i]):>0.1f}%\n", file=filename)

def top_k_mean(test_acc, k=5):
    test_acc = torch.tensor(test_acc)  
    topk_vals, _ = torch.topk(test_acc, k=k, dim=0)  
    return torch.mean(topk_vals, dim=0)  

if __name__ == '__main__':
    # Define record path
    root_path = '.'
    data_path = os.path.join(root_path, './Dataset').replace('\\', '/')
    save_path = f'Result/Results_IAVFL_2_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = open(f'Result/Results_IAVFL_2_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/A_training_process.txt', 'w+')
    
    if args.dataset == 'utkface' or args.dataset == 'celeba':
        train_data_nondataloader, test_data, num_classes, num_classes2, channel, hidden = load_data(args.dataset, args.attributes, data_path, args.batch_size)
    elif args.dataset == "imagenet":
        train_data_nondataloader, test_data, num_classes, channel, hidden = get_tinyimagenet_bothloader(batch_size=args.batch_size, shuffle=True, seed=args.seed, device=device, classes = args.classes)
    elif args.dataset =='MIMIC':
        train_data_nondataloader, test_data, input_dim1, input_dim2, input_dim3, input_dim4, num_classes = load_data_tabular_4clients(args.dataset, args.batch_size)
    else:
        train_data_nondataloader, test_data, num_classes, channel, hidden = gen_dataset(args.dataset, data_path, device = device)

    if args.dataset =='cifar10' or args.dataset =='imagenet' or args.dataset =='utkface':
        ImageModel_student= ImageModel_4clients(dataset= args.dataset, hidden=hidden, num_cutlayer=args.num_cutlayer, num_classes=num_classes, mode='standalone', device=device)
        client_student_model_1, client_student_model_2, client_student_model_3, client_student_model_4, server_student_model = ImageModel_student.GetModel()
        
        ImageModel_teacher = ImageModel_4clients(dataset= args.dataset, hidden=hidden, num_cutlayer=int(args.num_cutlayer/4), num_classes=num_classes, mode='cat', device = device)
        client_teacher_model_1, client_teacher_model_2, client_teacher_model_3, client_teacher_model_4, server_teacher_model = ImageModel_teacher.GetModel()
    elif args.dataset =='MIMIC':
        
        client_student_model_1, client_student_model_2, client_student_model_3, client_student_model_4, server_student_model = def_tabular_model_4clients(dataset= args.dataset, level= args.level, input_dim1=input_dim1, input_dim2=input_dim2, input_dim3=input_dim3, input_dim4=input_dim4, num_classes=num_classes, num_cutlayer=args.num_cutlayer, mode='standalone', device=device)

        client_teacher_model_1, client_teacher_model_2, client_teacher_model_3, client_teacher_model_4, server_teacher_model = def_tabular_model_4clients(dataset= args.dataset, level= args.level, input_dim1=input_dim1, input_dim2=input_dim2, input_dim3=input_dim3, input_dim4=input_dim4, num_classes=num_classes, num_cutlayer=int(args.num_cutlayer/4), mode='cat', device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer_student_client1 = torch.optim.Adam(client_student_model_1.parameters(), lr=args.lr, foreach=False)
    optimizer_student_client2 = torch.optim.Adam(client_student_model_2.parameters(), lr=args.lr, foreach=False)
    optimizer_student_client3 = torch.optim.Adam(client_student_model_3.parameters(), lr=args.lr, foreach=False)
    optimizer_student_client4 = torch.optim.Adam(client_student_model_4.parameters(), lr=args.lr, foreach=False)
    optimizer_student_server  = torch.optim.Adam(server_student_model.parameters(),   lr=args.lr, foreach=False)  

    train_loss = []
    train_acc = []
    
    if args.dataset =='cifar10':
        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    elif args.dataset =='MIMIC':
        missing_ratios = [0, 0.3, 0.5, 0.7, 1]
    
    test_acc = [[0] * len(missing_ratios) for _ in range(args.epochs)]
    test_loss = [[0] * len(missing_ratios) for _ in range(args.epochs)]

    # start training
    for t in range(args.epochs):
        print(f"Epoch {t + 1}\n-------------------------------", file=filename)

        if args.dataset == 'cifar10' or args.dataset == "imagenet" or args.dataset == 'utkface' or args.dataset =='MIMIC' or args.dataset == 'bank' or args.dataset == 'avazu':
            train_data = get_train(train_data_nondataloader, args.batch_size, args.seed, device) 

        train_client(train_data, client_student_model_1, client_student_model_2, \
            client_student_model_3, client_student_model_4, server_student_model, \
            client_teacher_model_1, client_teacher_model_2, client_teacher_model_3, \
            client_teacher_model_4, server_teacher_model, t)
        
        test_client(test_data, client_student_model_1, client_student_model_2, \
            client_student_model_3, client_student_model_4, server_student_model, t)
        
    print("Done!", file=filename)

    k = 5
    test_acc_top_K_mean = top_k_mean(test_acc, k=5)
    
    if args.dataset == 'cifar10':
        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    elif args.dataset == 'MIMIC':
        missing_ratios = [0, 0.3, 0.5, 0.7, 1]

    print(f"-----Top{k} result------", file=filename)

    for i, missing_ratio in enumerate(missing_ratios):
        print(f"Missing_ratio: {missing_ratio}", file=filename)
        print(f"top{k}avg--test_average_acc: {test_acc_top_K_mean[i]}", file=filename)
    
    #save
    np.save(f'Result/Results_IAVFL_2_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/train_acc.npy', train_acc) 
    np.save(f'Result/Results_IAVFL_2_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/train_loss.npy', train_loss)
    np.save(f'Result/Results_IAVFL_2_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/test_acc.npy', test_acc) 
    np.save(f'Result/Results_IAVFL_2_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/test_loss.npy', test_loss) 

    #plt
    x = np.arange(len(train_acc))
    
    plt.figure()
    plt.plot(x, train_acc, label='train_acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.savefig(f'Result/Results_IAVFL_2_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/A_train_acc.png')
    plt.close()

    plt.figure()
    plt.plot(x, train_loss, label='train_loss', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig(f'Result/Results_IAVFL_2_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/A_train_loss.png')
    plt.close()

    plt.figure()
    for i, missing_ratio in enumerate(missing_ratios):
        acc_i = [row[i] for row in test_acc]  
        plt.plot(x, acc_i, label=f'Missing ratio: {missing_ratio}')

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy")
    plt.legend()
    plt.savefig(f'Result/Results_IAVFL_2_4clients/Results_batchsize{args.batch_size}/learningRate{args.lr}/Seed{args.seed}/{args.dataset}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/A_test_acc.png')
    plt.close()


    
    
    
    
    

