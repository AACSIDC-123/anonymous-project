
import os
import pandas
import numpy as np
import random
import torch
import torch.utils.data
import PIL.Image as Image
from PIL import Image
from functools import partial
from parse import args
from torchvision import datasets, transforms
from typing import Any, Callable, List, Optional, Union, Tuple
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
import random
import os
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Subset

from torch.utils.data import Subset, ConcatDataset, random_split

os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
device = torch.device(f"cuda:0") 

def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
def gen_dataset(dataset, data_path, device):
    
    if dataset == 'mnist': 
        train_data = datasets.MNIST(root=data_path, train=True, download=True, transform=ToTensor(),)
        test_data = datasets.MNIST(root=data_path, train=False, download=True, transform=ToTensor(), )
        num_classes = 10
        shape_img = (28, 28)
        channel = 1
        hidden = 588    
    
    elif dataset == 'fmnist':
        train_data = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=ToTensor(),)
        test_data = datasets.FashionMNIST(root=data_path, train=False, download=True, transform=ToTensor(), )
        num_classes = 10
        shape_img = (28, 28)
        channel = 1
        hidden = 588
        
        
    elif dataset == 'cifar10':
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  
        ])

       
        train_data = datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=transform
        )
        test_data = datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=transform
        )
        num_classes = 10
        shape_img = (32, 32)
        channel = 3
        hidden = 768

    else:
        exit('unknown dataset')
    
    test_data = get_test(test_data, args.batch_size)
    return train_data, test_data,  num_classes, channel, hidden

def load_data(data, attr, data_path, batch_size):
    
    channel = 3
    hidden = 768
    if "_" in attr:
        attr = attr.split("_")
    num_classes, shape_img, train, test= prepare_dataset(data, attr, data_path)

    train_length = len(train)
    sample_number = min(batch_size * args.num_batch, train_length)
    train_indices = train.indices[:sample_number]  
    train = Subset(train.dataset, train_indices)
    
    test_data = get_test(test, batch_size)
    
    for images, labels in test_data:
        print("Batch image shape:", images.shape) 
        break

    num_classes1 = num_classes[0]
    if data == 'utkface':
        num_classes2 = num_classes[1]
    if data == 'celeba':
        num_classes2 = num_classes[1]

    return train, test_data, num_classes1, num_classes2, channel, hidden

def get_train(train_dataset, batch_size, seed, device):
    
    if args.dataset == 'cifar10':
        shuffle_set = seed != 42
    else:
        shuffle_set = True

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)  

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=shuffle_set,  
        generator=generator if shuffle_set else None  
        
    )
    return train_loader

def get_test(test_dataset, batch_size):
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              drop_last=True,
                                              shuffle=False)
    return test_loader 



def get_shadow(shadow_dataset, batch_size):
    shadow_loader = torch.utils.data.DataLoader(shadow_dataset,
                                              batch_size=batch_size,
                                              drop_last=True,
                                              shuffle=False)
    return shadow_loader 


def prepare_dataset(dataset, attr, root):
    dataset_name = str(dataset)
    num_classes, dataset = get_model_dataset(dataset, attr=attr, root=root)
    length = len(dataset)
    each_length = length//10
    shape_img = (32, 32)

    if dataset_name == 'utkface':
        train, test = torch.utils.data.random_split(dataset, [7*each_length,  length-7*each_length], generator=torch.Generator(device=args.device))
    if dataset_name == 'celeba':
        train, test = torch.utils.data.random_split(dataset, [7*each_length,  length-7*each_length], generator=torch.Generator(device=args.device))


    return num_classes, shape_img, train, test


def get_model_dataset(dataset_name,  attr, root):
    if dataset_name.lower() == "utkface":
        if isinstance(attr, list):
            num_classes = []
            for a in attr:
                if a == "age":
                    num_classes.append(117)
                elif a == "gender":
                    num_classes.append(2)
                elif a == "race":
                    num_classes.append(4)
                else:
                    raise ValueError("Target type \"{}\" is not recognized.".format(a))
        else:
            if attr == "age":
                num_classes = 117
            elif attr == "gender":
                num_classes = 2
            elif attr == "race":
                num_classes = 4
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(attr))

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            ])

        dataset = UTKFaceDataset(root=root, attr=attr, transform=transform)

    if dataset_name.lower() == "celeba":
        if isinstance(attr, list):
            num_classes = [4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 2, 2, 2, 2]
            # Male
            attr_list = [[35], [20], [18], [21], [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13],
                         [14], [15], [16], [17], [19], [22], [23], [24], [25], [26], [27], [28], [29], [30], [32], [33], [34], [31], [36], [37], [38],
                         [39]]
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        dataset = CelebA(root=root, attr_list=attr_list, target_type=attr, transform=transform)

    return num_classes, dataset


class UTKFaceDataset(torch.utils.data.Dataset):
    def __init__(self, root, attr: Union[List[str], str] = "gender", transform=None, target_transform=None)-> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.processed_path = os.path.join(self.root, 'UTKFace/processed/')
        self.files = os.listdir(self.processed_path)
        if isinstance(attr, list):
            self.attr = attr
        else:
            self.attr = [attr]

        self.lines = []
        for txt_file in self.files:
            txt_file_path = os.path.join(self.processed_path, txt_file)
            with open(txt_file_path, 'r') as f:
                assert f is not None
                for i in f:
                    image_name = i.split('jpg ')[0]
                    attrs = image_name.split('_')
                    if len(attrs) < 4 or int(attrs[2]) >= 4  or '' in attrs:
                        continue
                    self.lines.append(image_name+'jpg')

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index:int)-> Tuple[Any, Any]:
        attrs = self.lines[index].split('_')

        age = int(attrs[0])
        gender = int(attrs[1])
        race = int(attrs[2])

        image_path = os.path.join(self.root, 'UTKFace/raw/', self.lines[index]+'.chip.jpg').rstrip()
        image = Image.open(image_path).convert('RGB')
  
        target: Any = []
        for t in self.attr:
            if t == "age":
                target.append(age)
            elif t == "gender":
                target.append(gender)
            elif t == "race":
                target.append(race)
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform:
            image = self.transform(image)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return image, target


class CelebA(torch.utils.data.Dataset):
    base_folder = "celeba"
    def __init__(
            self,
            root: str,
            attr_list: str,
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        self.root = root
        self.transform = transform
        self.target_transform =target_transform
        self.attr_list = attr_list

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None)

        self.filename = splits[mask].index.values
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = Image.open(os.path.join(self.root, self.base_folder, "img_celeba", self.filename[index]))
        target: Any = []
        for t, nums in zip(self.target_type, self.attr_list):
            final_attr = 0
            for i in range(len(nums)):
                final_attr += 2 ** i * self.attr[index][nums[i]]
            target.append(int(final_attr))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

def split_data(dataset, X):
    
    if dataset == 'MIMIC':
        X_1 = X[:, :8]
        X_2 = X[:, 8:]
        
    elif dataset == 'avazu':
        X_1 = X[:, :39]
        X_2 = X[:, 39:]
    
    elif dataset == 'credit':
        X_1 = X[:, :37]
        X_2 = X[:, 37:]

    elif dataset == 'bank':
        X_1 = X[:, :29]
        X_2 = X[:, 29:]
    
    # image dataset 
    else:
        X = X.to(device)
        X_1 = X.to(device).clone().detach().to(device)
        X_2 = X.to(device).clone().detach().to(device)

        shape_1 = list(range(int(X_1.shape[-1]/2)))    
        shape_2 = list(range(int(X_1.shape[-1]/2), int(X_1.shape[-1])))

        index1 = torch.tensor(shape_1).to(device)
        index2 = torch.tensor(shape_2).to(device)
        
        # switch the positions of index1 and index2
        X_1 = X_1.index_fill(3, index2, 0).to(device)
        X_2 = X_2.index_fill(3, index1, 0).to(device)
    
    return X_1, X_2

# split a image into 4 parts
def split_data_4clients(dataset, X):
    
    if dataset == 'cifar10':

        X = X.to(device)
        B, C, H, W = X.shape
        assert H == 32 and W == 32, "The input image size must be 32x32"

        X_1 = torch.zeros_like(X)
        X_2 = torch.zeros_like(X)
        X_3 = torch.zeros_like(X)
        X_4 = torch.zeros_like(X)

        X_1[:, :, 0:16, 0:16] = X[:, :, 0:16, 0:16]   
        X_2[:, :, 0:16, 16:32] = X[:, :, 0:16, 16:32]
        X_3[:, :, 16:32, 0:16] = X[:, :, 16:32, 0:16] 
        X_4[:, :, 16:32, 16:32] = X[:, :, 16:32, 16:32] 
    
    elif dataset == 'MIMIC':
        X_1 = X[:, :4]
        X_2 = X[:, 4:8]
        X_3 = X[:, 8:12]
        X_4 = X[:, 12:]

    return X_1, X_2, X_3, X_4

def load_data_tabular(dataset, batch_size):
    if dataset == 'MIMIC':
        data = pd.read_csv('./Dataset/MIMIC3/mimic3d.csv')
        
        drop_cols = [
            'LOSgroupNum', 'hadm_id', 'AdmitDiagnosis',
            'AdmitProcedure', 'religion', 'insurance',
            'ethnicity', 'marital_status', 'ExpiredHospital',
            'LOSdays', 'gender', 'admit_type', 'admit_location']

        input_dim_1 = 8
        input_dim_2 = 7
        
        num_classes = 4
        
        X = data.drop(drop_cols, axis=1)
        y = data['LOSgroupNum']

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # set train_data to 45000 and test_data to 11500
        train_data = X[:45000]
        test_data = X[-11500:] 

        trainfeatures = torch.tensor(np.array(train_data).astype(float)).float()
        trainlabels = torch.tensor(np.array(y[:45000]).astype(int)).long()
        testfeatures = torch.tensor(np.array(test_data).astype(float)).float()
        testlabels = torch.tensor(np.array(y[-11500:]).astype(int)).long()

        # MIMIC
        dataset_train = Data.TensorDataset(trainfeatures, trainlabels)
        indices_0 = [i for i, (_, label) in enumerate(dataset_train) if label == 0]
        indices_1 = [i for i, (_, label) in enumerate(dataset_train) if label == 1]
        indices_2 = [i for i, (_, label) in enumerate(dataset_train) if label == 2]
        indices_3 = [i for i, (_, label) in enumerate(dataset_train) if label == 3]
        
        min_count = min(len(indices_0), len(indices_1), len(indices_2), len(indices_3))
        min_number = min(int((batch_size * args.num_batch)/4), min_count)
        
        balanced_indices_0 = sorted(indices_0)[:min_number]
        balanced_indices_1 = sorted(indices_1)[:min_number]
        balanced_indices_2 = sorted(indices_2)[:min_number]
        balanced_indices_3 = sorted(indices_3)[:min_number]
        index_train = sorted(balanced_indices_0 + balanced_indices_1 + balanced_indices_2 + balanced_indices_3)
        train_iter = Subset(dataset_train, index_train)
       

        dataset_test = Data.TensorDataset(testfeatures, testlabels)
        test_iter = Data.DataLoader(
          dataset=dataset_test, 
          batch_size=batch_size, 
          drop_last=True,
          shuffle=False,
        )


    elif dataset == 'avazu':
        # features
        categorical_columns = ['site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category','device_id', 'device_ip', 'device_model', 'C14', 'C17', 'C19', 'C20', 'C21']
        continuous_columns = ['C1', 'banner_pos', 'device_type', 'device_conn_type', 'C15', 'C16', 'C18']
        label = ['click']
        
        # 1.Prepross
        # make Preprocess = True at the first run
        Preprocess = False
        if Preprocess == True:

            train_data_file = 'train_10W.txt'
            test_data_file  = 'test_2W.txt'
            # train_data_file = 'train_20W.txt'
            # test_data_file = 'test_4W.txt'

            data_dir =  './Dataset/avazu/'
            train = pd.read_csv(os.path.join(data_dir, train_data_file))  
            test  = pd.read_csv(os.path.join(data_dir, test_data_file))   
            data  = pd.concat([train, test], axis=0)

            categorical_data = data[categorical_columns]
            continuous_data = data[continuous_columns]
            y = data[label]

            # Set embedding dimension
            num_components = 5  # Dimension of the embedding vector

            # List to store embedding vectors
            embedding_list = []

            # Iterate over each categorical feature
            for cat_col in categorical_columns:

                # 1. Label Encoding: Convert categorical values to integer indices
                lbe = LabelEncoder()
                data[cat_col] = lbe.fit_transform(data[cat_col])

                # 2. Prepare for Embedding Layer
                num_categories = len(data[cat_col].unique())  # Number of unique categories for the feature

                # Create an embedding layer (input: number of categories, output: embedding vector dimension)
                embedding_layer = nn.Embedding(num_categories, num_components).cuda()  # Place the embedding layer on GPU

                # Convert categorical feature to tensor, preparing for input into the embedding layer
                category_indices = torch.tensor(data[cat_col].values, dtype=torch.long).cuda()  # Move indices to GPU

                # Obtain the embedding vectors and keep them on GPU
                category_embedding = embedding_layer(category_indices).detach().cpu().numpy()  # Retrieve embeddings and convert to NumPy array

                # 3. Convert embedding vectors to DataFrame, keeping the original format
                embedding_df = pd.DataFrame(category_embedding, columns=[f'emb_{cat_col}_dim{j+1}' for j in range(num_components)])
                embedding_list.append(embedding_df)

            # 4. Merge all categorical feature embeddings into a final DataFrame
            final_embedding = pd.concat(embedding_list, axis=1)  # Concatenate along columns

            # Normalize continuous features
            scaler = MinMaxScaler(feature_range=(0, 1))  
            normalized_continuous_data = scaler.fit_transform(continuous_data)

            # Convert normalized continuous features to DataFrame
            normalized_continuous_df = pd.DataFrame(normalized_continuous_data, columns=[f'cont_{i+1}' for i in range(len(continuous_columns))])

            # Label
            y_df = pd.DataFrame(y, columns=['click'])
            y_df.reset_index(drop=True, inplace=True)

            # 2. Save---new.csv
            final_data = pd.concat([
                            final_embedding.iloc[:, :7 * num_components],  
                            normalized_continuous_df.iloc[:, :4],          
                            final_embedding.iloc[:, 7 * num_components:],  
                            normalized_continuous_df.iloc[:, 4:],          
                            y_df                                           
                            ], axis=1)
            final_data.to_csv(f'./Dataset/avazu/avazu_processed_data_100000.csv', index=False)

        num_components = 5
        # 3. load--new.csv
        data = pd.read_csv(f'./Dataset/avazu/avazu_processed_data_{2000*args.batch_size}.csv')
        input_dim_1 = 7*num_components + 4
        input_dim_2 = 7*num_components + 3
        num_classes = 2

        X = data.drop(label, axis=1)
        y = data[label]

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        train_data = X[:int(len(X) * 0.8)]
        test_data = X[int(len(X) * 0.8):] 

        trainfeatures = torch.tensor(np.array(train_data).astype(float)).float()
        trainlabels = torch.tensor(np.array(y[:int(len(X)*0.8)]).astype(int)).long().squeeze()
        
        
        testfeatures = torch.tensor(np.array(test_data).astype(float)).float()
        testlabels = torch.tensor(np.array(y[int(len(X)*0.8):]).astype(int)).long().squeeze()
        
        # Sample data
        num_train_0 = sum(i == 0 for i in trainlabels)
        num_train_1 = sum(i == 1 for i in trainlabels)

        min_num = min(num_train_0, num_train_1)  
        num_samples = int(min_num) * 2
  
        dataset_train = Data.TensorDataset(trainfeatures, trainlabels)
        indices_0 = [i for i, (_, label) in enumerate(dataset_train) if label == 0]
        indices_1 = [i for i, (_, label) in enumerate(dataset_train) if label == 1]
       
        min_count = min(len(indices_0), len(indices_1)) 
        min_number = min(int((batch_size * args.num_batch)/2), min_count)
        
        balanced_indices_0 = sorted(indices_0)[:min_number]
        balanced_indices_1 = sorted(indices_1)[:min_number]
        index_train = sorted(balanced_indices_0 + balanced_indices_1)
        
        train_iter = Subset(dataset_train, index_train)
        
        dataset_test = Data.TensorDataset(testfeatures, testlabels)
        
        indices_0 = [i for i, (_, label) in enumerate(dataset_test) if label == 0]
        indices_1 = [i for i, (_, label) in enumerate(dataset_test) if label == 1]
        
        min_count = min(len(indices_0), len(indices_1))  
        balanced_indices_0 = sorted(indices_0)[:min_count]
        balanced_indices_1 = sorted(indices_1)[:min_count]
        
        index_test = sorted(balanced_indices_0 + balanced_indices_1)
        test_dataset = Subset(dataset_test, index_test)
        
        test_iter = get_test(test_dataset, args.batch_size)
        

    elif dataset =='bank':
        data = pd.read_csv(f'./Dataset/bank/bank_onehot.csv')
        drop_cols = ['y']
        input_dim_1, input_dim_2, num_classes = 29,28,2
        
        X = data.drop(drop_cols, axis=1)
        y = data['y']
        
        train_data = X[:int(len(X) * 0.8)]
        test_data = X[int(len(X) * 0.8):] 
     
        numeric_attrs = ["job_admin._1", "job_admin._10", "job_admin._11", "job_admin._2", "job_admin._3","job_admin._4","job_admin._5","job_admin._6","job_admin._7","job_admin._8","job_admin._9", "loan_1", "loan_2",  "education_1","education_2","education_3","education_4","education_5","education_6","education_7", "housing_1","housing_2", "contact_cellular_1", "contact_cellular_2","nr.employed", "pdays",  "previous",  "duration", "campaign",
                         "month_apr_1", "month_apr_10", "month_apr_2", "month_apr_3","month_apr_4","month_apr_5","month_apr_6","month_apr_7","month_apr_8","month_apr_9", "marital_divorced_1","marital_divorced_2","marital_divorced_3", "poutcome_failure_1","poutcome_failure_2","poutcome_failure_3", "default_1","default_2", "day_of_week_fri_1", "day_of_week_fri_2", "day_of_week_fri_3","day_of_week_fri_4", "day_of_week_fri_5", "emp.var.rate", "euribor3m",   "age", "cons.price.idx",    "cons.conf.idx"]
        
        print("First 29 columns:", numeric_attrs[:input_dim_1])
        print("Last 28 columns:", numeric_attrs[input_dim_1:])
        
        label = ['y']
        trainfeatures = torch.tensor(np.array(train_data[numeric_attrs])).float()
        testfeatures = torch.tensor(np.array(test_data[numeric_attrs])).float()
        trainlabels = torch.tensor(np.array(y[:int(len(X)*0.8)]).astype(int)).long()
        testlabels  = torch.tensor(np.array(y[int(len(X)*0.8):]).astype(int)).long()

        # Sample data
        num_train_0 = sum(i == 0 for i in trainlabels)
        num_train_1 = sum(i == 1 for i in trainlabels)

        min_num = min(num_train_0, num_train_1)
        num_samples = int(min_num) * 2

        dataset_train = Data.TensorDataset(trainfeatures, trainlabels)
        indices_0 = [i for i, (_, label) in enumerate(dataset_train) if label == 0]
        indices_1 = [i for i, (_, label) in enumerate(dataset_train) if label == 1]
        
        min_count = min(len(indices_0), len(indices_1))  
        min_number = min(int((batch_size * args.num_batch)/2), min_count)
        
        balanced_indices_0 = sorted(indices_0)[:min_number]
        balanced_indices_1 = sorted(indices_1)[:min_number]
        index_train = sorted(balanced_indices_0 + balanced_indices_1)
        
        # training data is fixed and would not change the set
        train_iter = Subset(dataset_train, index_train)

        dataset_test = Data.TensorDataset(testfeatures, testlabels)
        
        indices_0 = [i for i, (_, label) in enumerate(dataset_test) if label == 0]
        indices_1 = [i for i, (_, label) in enumerate(dataset_test) if label == 1]
     
        min_count = min(len(indices_0), len(indices_1))  
        balanced_indices_0 = sorted(indices_0)[:min_count]
        balanced_indices_1 = sorted(indices_1)[:min_count]

        index_test = sorted(balanced_indices_0 + balanced_indices_1)
        test_dataset = Subset(dataset_test, index_test)
        
        test_iter = get_test(test_dataset, args.batch_size)

    return train_iter, test_iter, input_dim_1, input_dim_2, num_classes

def load_data_tabular_4clients(dataset, batch_size):
    
    assert dataset == 'MIMIC', f"Error: dataset must be 'MIMIC', but got '{dataset}'"

    data = pd.read_csv('./Dataset/MIMIC3/mimic3d.csv')
    
    drop_cols = [
        'LOSgroupNum', 'hadm_id', 'AdmitDiagnosis',
        'AdmitProcedure', 'religion', 'insurance',
        'ethnicity', 'marital_status', 'ExpiredHospital',
        'LOSdays', 'gender', 'admit_type', 'admit_location']
    input_dim_1 = 4
    input_dim_2 = 4
    input_dim_3 = 4
    input_dim_4 = 3
    
    num_classes = 4
    
    X = data.drop(drop_cols, axis=1)
    y = data['LOSgroupNum']
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # set train_data to 45000 and test_data to 11500
    train_data = X[:45000]
    test_data = X[-11500:] 
    trainfeatures = torch.tensor(np.array(train_data).astype(float)).float()
    trainlabels = torch.tensor(np.array(y[:45000]).astype(int)).long()
    testfeatures = torch.tensor(np.array(test_data).astype(float)).float()
    testlabels = torch.tensor(np.array(y[-11500:]).astype(int)).long()
    
    dataset_train = Data.TensorDataset(trainfeatures, trainlabels)
    indices_0 = [i for i, (_, label) in enumerate(dataset_train) if label == 0]
    indices_1 = [i for i, (_, label) in enumerate(dataset_train) if label == 1]
    indices_2 = [i for i, (_, label) in enumerate(dataset_train) if label == 2]
    indices_3 = [i for i, (_, label) in enumerate(dataset_train) if label == 3]
    
    min_count = min(len(indices_0), len(indices_1), len(indices_2), len(indices_3))
    min_number = min(int((batch_size * args.num_batch)/4), min_count)
    
    balanced_indices_0 = sorted(indices_0)[:min_number]
    balanced_indices_1 = sorted(indices_1)[:min_number]
    balanced_indices_2 = sorted(indices_2)[:min_number]
    balanced_indices_3 = sorted(indices_3)[:min_number]
    index_train = sorted(balanced_indices_0 + balanced_indices_1 + balanced_indices_2 + balanced_indices_3)
    train_iter = Subset(dataset_train, index_train)
    
    dataset_test = Data.TensorDataset(testfeatures, testlabels)
    test_iter = Data.DataLoader(
      dataset=dataset_test, 
      batch_size=batch_size, 
      drop_last=True,
      shuffle=False,
    )
    
    return train_iter, test_iter, input_dim_1, input_dim_2, input_dim_3, input_dim_4, num_classes

def get_tinyimagenet_bothloader(batch_size=16, shuffle=True, seed=args.seed, device=device, classes=5):
 
    random.seed(seed)
    torch.manual_seed(seed)

    generator = torch.Generator(device=device)

    transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.48024505, 0.4480726, 0.39754787), (0.2717199, 0.26526922, 0.27396977))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.48024505, 0.4480726, 0.39754787), (0.2717199, 0.26526922, 0.27396977))
    ])

    full_training = datasets.ImageFolder('./Dataset/tiny-imagenet-200/train', transform=transform_train)
    full_testing = datasets.ImageFolder('./Dataset/tiny-imagenet-200/val', transform=transform_test)

    selected_classes = full_training.classes[:classes]  
    train_indices = [i for i, (_, label) in enumerate(full_training) if full_training.classes[label] in selected_classes]
    tinyimagenet_training = Subset(full_training, train_indices)
    
    test_indices = [i for i, (_, label) in enumerate(full_testing) if full_testing.classes[label] in selected_classes]
    tinyimagenet_testing = Subset(full_testing, test_indices)

    num_classes = classes  
    channel = 3
    hidden = 768  
  
    test_data = DataLoader(
        tinyimagenet_testing, 
        batch_size=batch_size, 
        drop_last = True,
        shuffle=False, 
        generator=generator
    )

    return tinyimagenet_training, test_data, num_classes, channel, hidden


