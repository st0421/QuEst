import argparse
import scipy.io
import torch
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = '2'

from model import ft_net
from torchvision import datasets, transforms

#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--name', default='resnet50', type=str, help='save model path')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')

parser.add_argument('--test_dir',default='../1.datasets/Market-1501/estimator_data/',type=str, help='./test_data')

opts = parser.parse_args()
name = opts.name

data_dir = opts.test_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#####################################################################
benign_name = 'train_query_ori'
base_name = 'train_query_IFGSM_eps'
file_names = [benign_name] + [f'{base_name}{i}' for i in range(1, 17)]

results = {}

for fname in file_names:
    results[fname] = scipy.io.loadmat(f'{data_dir}mat/{fname}.mat')


results = {}
for fname in file_names:
    results[fname] = scipy.io.loadmat(f'{data_dir}mat/{fname}.mat')

all_file_names = ['gallery'] + file_names
image_datasets = {name: datasets.ImageFolder(os.path.join(data_dir, 'pytorch', name)) for name in all_file_names}

query_features = {
    f'query_feature_{i}': torch.FloatTensor(results[i][f'query_f_{file_names[i]}'])
    for i in range(len(file_names))
}
query_features = {
    key: value.to(device) 
    for key, value in query_features.items()
}

query_labels = [r['query_label'][0] for r in results]
query_cams = [r['query_cam'][0] for r in results]

gallery_feature = torch.FloatTensor(results[benign_name]['gallery_f'])
gallery_cam = results[benign_name]['gallery_cam'][0]
gallery_label = results[benign_name]['gallery_label'][0]

gallery_feature = gallery_feature.to(device)

#######################################################################
# sort the images
def scoring(qf,gf):
    query = qf.view(-1,1)
    gf = gf.view(1,-1)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    return score    

def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1,1)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    index = np.argsort(score) 
    index = index[::-1]
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) 

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index, score[index]

def sort_img_rank(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1,1)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    index = np.argsort(score)
    index = index[::-1]
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    junk_index1 = np.argwhere(gl==-1)   
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) 

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index, score[index]
    
########################################################################

data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_network(network):
    save_path = os.path.join('./model',name,'ft_%s.pth'%opts.which_epoch)
    network.load_state_dict(torch.load(save_path))

    return network

Q_R_score = []
AR_err = []
imgs=[]
temp = ""
query_response = []
label = []

with open("data/metricfgsm.txt",'a') as f:
    for idx, file_name in enumerate(file_names):
        print(f"start process: {file_name}")
        for i in range(len(query_labels[file_name])):
            query_path, _ = image_datasets[file_name].imgs[i]
            index, score = sort_img(query_features[file_name][i],query_labels[file_name][i],query_cams[file_name][i],gallery_feature,gallery_label,gallery_cam)
            index_query_rank1, _ = sort_img_rank(gallery_feature[index[0]],gallery_label[index[0]],gallery_cam[index[0]],gallery_feature,gallery_label,gallery_cam)
            
            Q_R_score = [round(score[r], 4) for r in range(10)]
            
            for q_idx in range(10):
                for a in range(q_idx+1, 10):
                    score_between_query_ranked = scoring(gallery_feature[index[q_idx]],gallery_feature[index[a]])
                    Q_R_score.append(round(score_between_query_ranked[0],4))

            for t in Q_R_score:
                temp += str(t)+" "

            temp = ' '.join(map(str, Q_R_score)) + f'\t{idx}\n'

            f.write(temp)
            Q_R_score.clear()

        print(f"end process: {idx}")