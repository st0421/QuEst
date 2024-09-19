import os
import warnings
from tqdm import tqdm
import argparse

from PIL import Image
import numpy as np
import pandas as pd
import pickle
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn.functional as F
from torch.nn import init
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data', help='Raf-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs.')
    parser.add_argument('--data', type=str, default='QRA_metricfgsm.txt')
    parser.add_argument('--input_dim', type=int, default=57)

    return parser.parse_args()

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class ADDataSet(Dataset):
    def __init__(self, path='data', file=''):
        self.path = path
        self.file = file

        df = pd.read_csv(os.path.join(self.path, self.file), sep='\t', header=None,names=['QRA','label'])
        
        selected_columns = df[['QRA']]

        self.data = selected_columns.apply(lambda row: [float(value) for value in row.values[0].split()], axis=1).tolist()
        
        self.label = df[['label']].values

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        print(f' distribution of samples: {self.sample_counts}')

    def __len__(self):
        return sum(self.sample_counts)
    
    def get_labels(self):
        return self.label
    
    def __getitem__(self, idx):
        data = self.data[idx]
        new_data = data.copy()
        new_data.append(np.std(data[:10]))
        new_data.append(np.std(data[10:]))
        new_data = torch.tensor(new_data, dtype=torch.float32)
        label = self.label[idx].astype(np.float32)
        
        return new_data, label[0]

class MLP(nn.Module):
    def __init__(self,input_dim=57,class_num=1,num_bottleneck=256,droprate=0.5):
        super(MLP,self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, num_bottleneck).apply(weights_init_kaiming)
        self.bn1 = nn.BatchNorm1d(num_bottleneck).apply(weights_init_kaiming)
        self.relu = nn.LeakyReLU(0.1).apply(weights_init_kaiming)
        self.fc3 = nn.Linear(num_bottleneck, num_bottleneck).apply(weights_init_kaiming)
        self.bn3 = nn.BatchNorm1d(num_bottleneck).apply(weights_init_kaiming)
        self.relu3 = nn.LeakyReLU(0.1).apply(weights_init_kaiming)
        self.fc2 = nn.Linear(num_bottleneck, class_num).apply(weights_init_classifier)        

    def forward(self,x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)        
        x = self.fc2(x)
        return x
        
def run_training():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = MLP(input_dim=args.input_dim).to(device)

    dataset = ADDataSet(args.path, args.data)    
    dataset_length = len(dataset)
    train_size = int(len(dataset)*0.8)
    val_size = dataset_length - train_size
    train_dataset , val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Training Data Size : {len(train_dataset)}")
    print(f"Validation Data Size : {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, 
                                batch_size = args.batch_size,
                                num_workers = args.workers,
                                shuffle = True,  
                                pin_memory = True)

    val_loader = DataLoader(val_dataset,
                            batch_size = args.batch_size,
                            num_workers = args.workers,
                            shuffle = False,  
                            pin_memory = True)
    
    criterion_cls = torch.nn.BCELoss().to(device) 
    params = list(model.parameters())
    optimizer = torch.optim.SGD(params,lr=args.lr, weight_decay = 1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()
        for (data, targets) in train_loader:
            iter_cnt += 1
            optimizer.zero_grad()

            data = data.to(device)
            targets = targets.to(device) 
            targets = targets.float()
            out = model(data)
            out = out.squeeze()

            loss = criterion_cls(out,targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss
            predicts = torch.gt(out, torch.ones_like(out)/2)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        tqdm.write('[Epoch %d] Train acc: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss,optimizer.param_groups[0]['lr']))
        
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            baccs = []
            train_pred_list = []
            train_labels_list = []
            model.eval()
            for (data, targets) in val_loader:
                data = data.to(device)
                targets = targets.to(device)
                targets = targets.float()
                out = model(data)

                train_pred_list.append(out.cpu().flatten())
                train_labels_list.append(targets.cpu().numpy().flatten())

                out = out.squeeze()
                loss = criterion_cls(out,targets)
                running_loss += loss
                iter_cnt+=1
                
                predicts = torch.gt(out, torch.ones_like(out)/2)
                
                correct_num  = torch.eq(predicts,targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)

                train_labels = np.concatenate(train_labels_list)
                train_pred = np.concatenate(train_pred_list)
                auc_score = roc_auc_score(train_labels, train_pred)
                train_labels_list = []
                train_pred_list = []
                baccs.append(balanced_accuracy_score(targets.cpu().numpy(),predicts.cpu().numpy()))
                
            running_loss = running_loss/iter_cnt   
            scheduler.step()

            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(),4)

            best_acc = max(acc,best_acc)


            bacc = np.around(np.mean(baccs),4)
            tqdm.write("[Epoch %d] Val acc:%.4f. auc_score:%.4f. bacc:%.4f. Loss:%.3f." % (epoch, acc,auc_score, bacc, running_loss))
            tqdm.write("best_acc:" + str(best_acc))

if __name__ == "__main__":        
    run_training()