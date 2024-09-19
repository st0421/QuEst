# -*- coding: utf-8 -*-

from __future__ import print_function, division

from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from torch.utils import data

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import time
import os
import scipy.io
import yaml
import math
from model import ft_net, Backbone_nFC
from utils import fuse_all_conv_bn
from reid_dataset import import_MarketDuke_nodistractors
from reid_dataset import import_Market1501Attribute_binary
from reid_dataset import import_DukeMTMCAttribute_binary
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='../1.datasets/',type=str, help='./test_data')
parser.add_argument('--name', default='resnet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader) # for the new pyyaml via 'conda install pyyaml'
opt.stride = config['stride']
if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 751 

str_ids = opt.gpu_ids.split(',')
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------

h, w = 256, 128

data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
data_dir = test_dir


class Test_Dataset(data.Dataset):
    def __init__(self, data_dir, dataset_name, transforms=None, query_gallery='all' ):
        train, query, gallery = import_MarketDuke_nodistractors(data_dir, dataset_name)

        if dataset_name == 'Market-1501':
            self.train_attr, self.test_attr, self.label = import_Market1501Attribute_binary(data_dir)
        elif dataset_name == 'DukeMTMC-reID':
            self.train_attr, self.test_attr, self.label = import_DukeMTMCAttribute_binary(data_dir)
        else:
            print('Input should only be Market1501 or DukeMTMC')

        if query_gallery == 'query':
            self.test_data = query['data']
            self.test_ids = query['ids']
        elif query_gallery == 'gallery':
            self.test_data = gallery['data']
            self.test_ids = gallery['ids']
        elif query_gallery == 'all':
            self.test_data = gallery['data'] + query['data']
            self.test_ids = gallery['ids']
        else:
            print('Input shoud only be query or gallery;')

        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.test_data[index][0]
        id = self.test_data[index][2]
        label = np.asarray(self.test_attr[id])
        data = Image.open(img_path)
        data = self.transforms(data)
        name = self.test_data[index][4]
        return data, label, id, name

    def __len__(self):
        return len(self.test_data)

    def labels(self):
        return self.labels



image_datasets = {x: Test_Dataset(data_dir, dataset_name='Market-1501',transforms=data_transforms, query_gallery=x) for x in ['gallery','query']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                            shuffle=False, num_workers=0) for x in ['gallery','query']}



######################################################################
# Load model
#---------------------------
def load_network(ReIDnetwork):
    ReID_save_path = os.path.join('./model',name,'ReID_net_%s.pth'%opt.which_epoch)
    ReIDnetwork.load_state_dict(torch.load(ReID_save_path))
    return ReIDnetwork

######################################################################
# Extract feature
# ----------------------
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    #features = torch.FloatTensor()
    count = 0
    if opt.linear_num <= 0:
        opt.linear_num = 2048
    camera_id = []
    labels = []
    for iter, data in enumerate(dataloaders):

        img, label, ids, name = data
        for id in ids:
            labels.append(int(id))

        for cn in name:
            camera = cn.split('c')[1]   
            camera_id.append(int(camera[0]))
        
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n,opt.linear_num).zero_().cuda()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img)
                ff += outputs
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        
        if iter == 0:
            features = torch.FloatTensor( len(dataloaders.dataset), ff.shape[1])
        start = iter*opt.batchsize
        end = min( (iter+1)*opt.batchsize, len(dataloaders.dataset))
        features[ start:end, :] = ff
    return features, camera_id, labels

# Load Collected data Trained model
print('-------Load ReID model-----------')
ReID_model_structure= ft_net(opt.nclasses, stride = opt.stride, ibn = False, linear_num=opt.linear_num)

ReIDmodel = load_network(ReID_model_structure)
ReIDmodel.classifier.classifier = nn.Sequential()

# Change to test mode
ReIDmodel = ReIDmodel.eval()

if torch.cuda.is_available():
    ReIDmodel = ReIDmodel.cuda()

model = fuse_all_conv_bn(ReIDmodel)

# Extract feature
since = time.time()
with torch.no_grad():
    gallery_feature, gallery_cam, gallery_label = extract_feature(model,dataloaders['gallery'])
    query_feature, query_cam, query_label = extract_feature(model,dataloaders['query'])
    
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.2f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam, 'query_f_adv':query_feature.numpy(), 'query_label':query_label, 'query_cam':query_cam}
scipy.io.savemat('%s_adv.mat'%name,result)
