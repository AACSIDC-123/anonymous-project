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

# Define A random_seed
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
    
# Train_Server Side Program
def train_server(client3_fx, Y_1, t, batch_id, correct, server_model, num_All, size):
    server_model.train()
    correct = correct

    _, server_result = server_model(client3_fx)
    loss_ce = criterion(server_result, Y_1) 

    loss = loss_ce 
    
    optimizer_server.zero_grad()
    loss.backward()
    dfx3 = client3_fx.grad.clone().detach().to(device)
    optimizer_server.step()
    
    correct += (server_result.argmax(1) == Y_1).type(torch.float).sum().item()

    if (t+1) %10 == 0 or t == args.epochs-1:
        if batch_id == args.num_batch -1:
            save_server_path = f'Result/Results_standalone3_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/server_epoch{t+1}.pth'
            torch.save(server_model.state_dict(), save_server_path)
        
    if batch_id == args.num_batch-1:
        correct_train_ratio = correct / (args.num_batch*num_All)
        loss, current = loss.item(), (batch_id + 1) *args.batch_size
        print(f"Train-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train_ratio):>0.1f}%", file=filename)
        train_acc.append(100 * correct_train_ratio)
        train_loss.append(loss)

    return dfx3, correct

# Train_Client Side Program
def train_client(dataloader, client_model_3, server_model, t):
    client_model_3.train()

    correct = 0

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
        X_3_overlap   = X_3[:args.num_overlap]
        X_3_complete1 = X_3[args.num_overlap : int(((args.batch_size - args.num_overlap)/4)*1 + args.num_overlap)]
        X_3_complete2 = X_3[int(((args.batch_size - args.num_overlap)/4)*1 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*2 + args.num_overlap)]
        X_3_complete3 = X_3[int(((args.batch_size - args.num_overlap)/4)*2 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*3 + args.num_overlap)]
        X_3_complete4 = X_3[int(((args.batch_size - args.num_overlap)/4)*3 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*4 + args.num_overlap)]

        X_3_All = torch.cat([X_3_overlap, X_3_complete1, X_3_complete2, X_3_complete4], dim=0)
        
        Y_overlap = Y_1[:args.num_overlap]
        Y_complete1 = Y_1[args.num_overlap : int(((args.batch_size - args.num_overlap)/4)*1 + args.num_overlap)]
        Y_complete2 = Y_1[int(((args.batch_size - args.num_overlap)/4)*1 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*2 + args.num_overlap)]
        Y_complete3 = Y_1[int(((args.batch_size - args.num_overlap)/4)*2 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*3 + args.num_overlap)]
        Y_complete4 = Y_1[int(((args.batch_size - args.num_overlap)/4)*3 + args.num_overlap) : int(((args.batch_size - args.num_overlap)/4)*4 + args.num_overlap)]
        
        Y_All = torch.cat([Y_overlap, Y_complete1, Y_complete2, Y_complete4], dim=0)
        
        num_complete1 = int((args.batch_size - args.num_overlap)/4)
        num_complete2 = int((args.batch_size - args.num_overlap)/4)
        num_complete3 = int((args.batch_size - args.num_overlap)/4)
        num_complete4 = int((args.batch_size - args.num_overlap)/4)
        
        num_All = args.num_overlap + num_complete1 + num_complete2 + num_complete4
        
        # client1--train and update
        fx3 = client_model_3(X_3_All)
        # Sending activations to server and receiving gradients from server
        client3_fx = (fx3).clone().detach().requires_grad_(True)
        g_fx3, correct = train_server(client3_fx, Y_All, t, batch_id, correct, server_model, num_All, size)
        
        optimizer_client3.zero_grad()
        (fx3).backward(g_fx3)
        optimizer_client3.step()
        
        # record 
        if (t+1) %10 ==0 or t == args.epochs-1:
            if batch_id == args.num_overlap-1:  
                # save_path1 = f'Result/Results_standalone3_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client1_epoch{t+1}.pth'
                save_path3 = f'Result/Results_standalone3_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/client3_epoch{t+1}.pth'
                # torch.save(client_model_1.state_dict(), save_path1)
                torch.save(client_model_3.state_dict(), save_path3)
            
        if batch_id >= args.num_batch -1:
            break
        

def test_server(dataloader, client_3_embeddings, Y_1, batch_id, correct, loss, server_model, size, t):
    
    server_model.eval()
    correct = correct
    loss = loss

    for i, client_3_embedding in enumerate(client_3_embeddings):
        
        optimizer_server.zero_grad()
        _, fx_server = server_model(client_3_embedding)
        loss[i] += criterion(fx_server, Y_1).item()
        correct[i] += (fx_server.argmax(1) == Y_1).type(torch.float).sum().item()

    current = (batch_id + 1) * len(Y_1)

    if batch_id == len(dataloader) - 1:
        
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
            print(f"Missing ratio is: {missing_ratio}", file=filename)
            print(f"Test-loss: {test_loss[t][i]:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * test_acc[t][i]):>0.1f}%\n", file=filename)

    return correct, loss


def test_client(dataloader, client_model_3, server_model, t):

    client_model_3.eval()

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
            Y_2 = target[1].view(-1, 1).to(device)
        else:
            Y_1 = target.to(device)

        if args.dataset =='cifar10':
            missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        elif args.dataset =='MIMIC':
            missing_ratios = [0, 0.3, 0.5, 0.7, 1]
            
        client_3_embeddings = [] 
        
        half_width = int(X.shape[-1]/2)
        half_height = int(X.shape[-2]/2)
        
        for i, missing_ratio in enumerate(missing_ratios):
            
            if missing_ratio > 0:
                if args.dataset == 'cifar10':
                    # Client 3
                    X3_blank = X_3.clone()
                    X3_blank[:, :, :int(half_height * (1 + missing_ratio)), int(half_width * (1 - missing_ratio)):] = 0
                elif args.dataset == 'MIMIC':
                    Missing_item_1, Missing_item_2, Missing_item_3, Missing_item_4 = get_Missing_item_MIMIC(missing_ratio)
                    X3_blank = X_3.clone()
                    X3_blank[:,-Missing_item_3:] = 0
            
            else:
                X3_blank = X_3

            fx3 = client_model_3(X3_blank)
            client_3_embeddings.append(fx3.clone().detach().requires_grad_(True))
            
        # Sending activations to server and receiving gradients from server
        correct, loss = test_server(dataloader, client_3_embeddings, Y_1, batch_id, correct, loss, server_model, size, t)

def top_k_mean(test_acc, k=5):
    test_acc = torch.tensor(test_acc)  
    topk_vals, _ = torch.topk(test_acc, k=k, dim=0)  # shape: [k, 7]
    return torch.mean(topk_vals, dim=0)  # shape: [7,]

if __name__ == '__main__':
    # Define record path
    root_path = '.'
    data_path = os.path.join(root_path, './Dataset').replace('\\', '/')
    save_path = f'Result/Results_standalone3_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = open(f'Result/Results_standalone3_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/A_training_process.txt', 'w+')
    
    if args.dataset == 'utkface' or args.dataset == 'celeba':
        train_data_nondataloader, test_data, num_classes, num_classes2, channel, hidden = load_data(args.dataset, args.attributes, data_path, args.batch_size)
    elif args.dataset == "imagenet":
        train_data_nondataloader, test_data, num_classes, channel, hidden = get_tinyimagenet_bothloader(batch_size=args.batch_size, shuffle=True, seed=args.seed, device=device, classes = args.classes)
    elif args.dataset =='MIMIC':
        train_data_nondataloader, test_data, input_dim1, input_dim2, input_dim3, input_dim4, num_classes = load_data_tabular_4clients(args.dataset, args.batch_size)
    else:
        train_data_nondataloader, test_data, num_classes, channel, hidden = gen_dataset(args.dataset, data_path, device = device)

    if args.dataset =='cifar10' or args.dataset =='imagenet' or args.dataset =='utkface':
        ImageModel_init= ImageModel_4clients(dataset= args.dataset, hidden=hidden, num_cutlayer=args.num_cutlayer, num_classes=num_classes, mode='standalone', device=device)
        client_model_1, client_model_2, client_model_3, client_model_4, server_model = ImageModel_init.GetModel()
    elif args.dataset =='MIMIC':
        
        client_model_1, client_model_2, client_model_3, client_model_4, server_model = def_tabular_model_4clients(dataset= args.dataset, level= args.level, input_dim1=input_dim1, input_dim2=input_dim2, input_dim3=input_dim3, input_dim4=input_dim4, num_classes=num_classes, num_cutlayer=args.num_cutlayer, mode='standalone', device=device)

    # criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_client3 = torch.optim.Adam(client_model_3.parameters(), lr=args.lr, foreach=False)
    optimizer_server  = torch.optim.Adam(server_model.parameters(),   lr=args.lr, foreach=False)  

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
        print(f"\nEpoch {t + 1}\n-------------------------------", file=filename)
        if args.dataset == 'cifar10' or args.dataset == "imagenet" or args.dataset == 'utkface' or args.dataset =='MIMIC' or args.dataset == 'bank' or args.dataset == 'avazu':
            train_data = get_train(train_data_nondataloader, args.batch_size, args.seed, device) 

        train_client(train_data, client_model_3, server_model, t)
        test_client(test_data, client_model_3, server_model, t)
    print("Done!", file=filename)


    k = 5
    test_acc_top_K_mean = top_k_mean(test_acc, k=5)
    
    if args.dataset =='cifar10':
        missing_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    elif args.dataset =='MIMIC':
        missing_ratios = [0, 0.3, 0.5, 0.7, 1]

    print(f"-----Top{k} result------", file=filename)

    for i, missing_ratio in enumerate(missing_ratios):
        print(f"Missing_ratio: {missing_ratio}", file=filename)
        print(f"top{k}avg--test_average_acc: {test_acc_top_K_mean[i]}", file=filename)
    
    #save
    np.save(f'Result/Results_standalone3_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/train_acc.npy', train_acc) 
    np.save(f'Result/Results_standalone3_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/train_loss.npy', train_loss)
    np.save(f'Result/Results_standalone3_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/test_acc.npy', test_acc) 
    np.save(f'Result/Results_standalone3_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/test_loss.npy', test_loss) 

    #plt
    x = np.arange(len(train_acc))

    plt.figure()
    plt.plot(x, train_acc, label='train_acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.savefig(f'Result/Results_standalone3_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/A_train_acc.png')
    plt.close()

    plt.figure()
    plt.plot(x, train_loss, label='train_loss', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig(f'Result/Results_standalone3_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/A_train_loss.png')
    plt.close()
    
    plt.figure()
    for i, missing_ratio in enumerate(missing_ratios):
        acc_i = [row[i] for row in test_acc] 
        plt.plot(x, acc_i, label=f'Missing ratio: {missing_ratio}')

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy")
    plt.legend()
    plt.savefig(f'Result/Results_standalone3_4clients/Results_batchsize{args.batch_size}/Seed{args.seed}/nonoverlap1_ratio_{args.nonoverlap1_ratio}/{args.dataset}/iclasses{args.classes}/{args.model}-level{args.level}/client{args.number_client}-c{args.num_cutlayer}/samples{args.num_batch*args.batch_size}/overlap{args.num_overlap*args.num_batch}/A_test_acc.png')
    plt.close()

