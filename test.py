import os
import warnings
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn import init


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data', help='dataset path.')
    parser.add_argument('--data', type=str, default='QRA_metricfgsm.txt')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs.')
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
    

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count

def run_validation():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        
    model = MLP(input_dim=args.input_dim).to(device)

    val_dataset = ADDataSet(args.path, args.data)    

    print(f"Validation Data Size : {len(val_dataset)}")

    val_loader = DataLoader(val_dataset,
                            batch_size = args.batch_size,
                            num_workers = args.workers,
                            shuffle = False,  
                            pin_memory = True)
    checkpoint = torch.load('./weights/QuEst_pretrained.pth')
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    model.to(device)
    criterion_mae = torch.nn.L1Loss().to(device)
    params = list(model.parameters())
    optimizer = torch.optim.SGD(params,lr=args.lr, weight_decay = 1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    loss_meter = AverageMeter()
    error_meter = AverageMeter()

    best_err = 100

    with torch.no_grad():
        with open('QRA_metricfgsm.txt','a') as f:
            all_targets = []
            all_predicts = []
            sample_cnt = 0
            bingo_cnt = 0


            model.eval()
            for (data, targets, index) in val_loader:
                data = data.to(device)
                targets = targets.to(device)
                out = model(data)

                loss = criterion_mae(out,targets)
                error = criterion_mae(out,targets)

                predicts = out.to(torch.int)

                all_targets.append(targets[0].cpu().detach().numpy())
                all_predicts.append(predicts[0].cpu().detach().numpy())
                
                pred = torch.where(predicts.cpu())

                correct_num  = torch.eq(pred,targets.cpu())
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)
                num = data.size(0)
                loss_meter.update(loss.item(), num)
                error_meter.update(error.item(), num)

            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(),4)
            scheduler.step()
            best_err = min(error_meter.val,best_err)
            tqdm.write('Val: Error: %.4f. Loss: %.3f. Acc: %.4f' % (error_meter.avg, loss_meter.avg, acc))
        
if __name__ == "__main__":        
    run_validation()