# %%
import os
import argparse
import logging
import sys
import datetime
import yaml
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx

import math
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.nn import PairNorm

# %%
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### 训练参数
def get_args():
    parser = argparse.ArgumentParser('gcn parameters')
    parser.add_argument('--data', type=str, default='cora', choices=['cora', 'citeseer', 'ppi'], help='dataset to use')
    parser.add_argument('--cpu', action='store_true', help='wheather to use cpu , if not set, use gpu')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--hidden', type=int, default=16, help='number of hidden units')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--saved_dir', type=str, default='./saved_dir')
    parser.add_argument('--pairnorm', action='store_true', help='whether to use pairnorm')
    parser.add_argument('--dropedge', action='store_true', help='whether to use dropedge')
    parser.add_argument('--activate', type=str, default='relu', choices=['relu', 'tanh'], help='activation function')

    args = parser.parse_args()

    if not args.cpu and not torch.cuda.is_available():
        logger.error('cuda is not available, try running on CPU by adding param --cpu')
        sys.exit()

    args.begin_time = datetime.datetime.now().strftime('%H%M%S')
    args.saved_dir = Path(args.saved_dir) / args.begin_time
    args.saved_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f'Save Dir : {str(args.saved_dir)}')    

    with open(args.saved_dir / 'opt.yaml','w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    return args

args = get_args()
device = torch.device('cuda' if not args.cpu else 'cpu')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if not args.cpu:
    torch.cuda.manual_seed(args.seed)

# %%
### 处理cora数据
def normalize(mx):
    rowsum = np.sum(mx, axis=1)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_max_inv = np.diag(r_inv)
    mx = r_max_inv.dot(mx)
    return mx

def load_cora():
    data_dir = 'cora_data'

    nodes = pd.read_csv(os.path.join(data_dir, 'cora.content'), sep='\t', header=None)
    edges = pd.read_csv(os.path.join(data_dir, 'cora.cites'), sep='\t', header=None)
    nodes_num = nodes.shape[0]

    features = nodes.iloc[:, 1:-1].values.astype(float)
    labels = pd.get_dummies(nodes[len(nodes.columns)-1])
    
    map = dict(zip(list(nodes[0]), list(nodes.index)))
    adj = np.zeros((nodes_num, nodes_num))
    for i, j in zip(edges[0], edges[1]):
        x = map[i]; y = map[j]
        adj[x][y] = adj[y][x] = 1
    
    features = normalize(features)
    adj = normalize(adj + np.eye(adj.shape[0]))

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])
    adj = torch.FloatTensor(adj)

    train_idx = torch.LongTensor(range(2000))
    valid_idx = torch.LongTensor(range(2000, 2300))
    test_idx = torch.LongTensor(range(2300, features.shape[0]))

    return adj, features, labels, train_idx, valid_idx, test_idx

# %%
### 处理citeseer数据
def load_citeseer():
    data_dir = 'citeseer_data'

    nodes = pd.read_csv(os.path.join(data_dir, 'citeseer.content'), sep='\t', header=None, dtype={0:str})
    edges = pd.read_csv(os.path.join(data_dir, 'citeseer.cites'), sep='\t', header=None, dtype=str)
    nodes_num = nodes.shape[0]

    features = nodes.iloc[:, 1:-1].values.astype(float)
    labels = pd.get_dummies(nodes[len(nodes.columns)-1])
    
    map = dict(zip(list(nodes[0]), list(nodes.index)))
    adj = np.zeros((nodes_num, nodes_num))
    for i, j in zip(edges[0], edges[1]):
        if i not in map or j not in map:
            continue
        x = map[i]; y = map[j]
        adj[x][y] = adj[y][x] = 1
    
    features = normalize(features)
    adj = normalize(adj + np.eye(adj.shape[0]))

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])
    adj = torch.FloatTensor(adj)

    train_idx = torch.LongTensor(range(2500))
    valid_idx = torch.LongTensor(range(2500, 3000))
    test_idx = torch.LongTensor(range(3000, features.shape[0]))

    return adj, features, labels, train_idx, valid_idx, test_idx

# %%
### 处理ppi数据
def load_ppi():
    # TODO
    ppi_dir = 'ppi_data'
    dataset = PPI(ppi_dir)
    train_data, val_data, test_data = dataset[0]
    nodes, edges = train_data[0], train_data[1]
    nodes_num = nodes.shape[0]
    adj = np.zeros((nodes_num, nodes_num))
    for i in range(edges.shape[1]):
        x = edges[0][i]; y = edges[1][i]
        adj[x][y] = adj[y][x] = 1

    features = normalize(nodes.numpy())
    adj = normalize(adj + np.eye(adj.shape[0]))

    print(features.shape, adj.shape)

# %%
### 加载数据
def load_data():
    if(args.data == "cora"):
        return load_cora()
    elif(args.data == 'citeseer'):
        return load_citeseer()
    elif(args.data == 'ppi'):
        return load_ppi()

# %%
adj, features, labels, train_idx, valid_idx, test_idx = load_data()
adj = adj.to(device)
features = features.to(device)
labels = labels.to(device)
train_idx = train_idx.to(device)
valid_idx = valid_idx.to(device)
test_idx = test_idx.to(device)

# %%
### 模型
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, activate = 'relu', do_pairnorm=False, do_dropedge=False):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

        self.activate = F.relu if activate == 'relu' else torch.tanh
        
        self.do_pairnorm = do_pairnorm
        self.do_dropedge = do_dropedge
        self.pairnorm = PairNorm()

    def forward(self, x, adj):
        if self.do_dropedge:
            mask = torch.rand(adj.shape).cuda() > 0.8
            adj=adj * mask
        x = self.gc1(x, adj)
        if self.do_pairnorm:
            x = self.pairnorm(x)
        x = self.activate(x)
        #x = F.tanh(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

# %%
### 构建模型
model = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item()+1, dropout=args.dropout, activate=args.activate, do_pairnorm=args.pairnorm, do_dropedge=args.dropedge).to(device)
model_parameters = filter(lambda p:p.requires_grad,model.parameters())
n_params = sum([p.numel() for p in model_parameters])
logger.info('Model Setting ...')
logger.info(f'Number of model params: {n_params}')

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

criterion = F.nll_loss

# %%
### 保存模型
def save_checkpoint(checkpoint_path, model, optimizer, epoch, best_acc):
    if isinstance(model,(torch.nn.parallel.DistributedDataParallel,torch.nn.DataParallel)):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    checkpoint = {
        'modelstate':model_state_dict,
        'optimstate':optimizer.state_dict(),
        'epoch_id':epoch,
        'best_acc':best_acc
        }
    torch.save(checkpoint, checkpoint_path)

# %%
### 保存损失和准确率
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False
def save_figure(target, title, train_record, valid_record):
    plt.figure()
    plt.plot(np.arange(len(train_record)), train_record, label='train')
    plt.plot(np.arange(len(valid_record)), valid_record, label='valid')
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.savefig(target)
    plt.close()

### 计算准确率
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# %%
### 训练过程
def train():
    best_acc = 0
    early_stop_count = 0

    train_loss_record = []
    valid_loss_record = []
    train_acc_record = []
    valid_acc_record = []

    try:
        for epoch in range(0, args.epoch):
            model.train()
            optimizer.zero_grad()
            output = model(features, adj)
            loss_train = criterion(output[train_idx], labels[train_idx])
            acc_train = (output[train_idx].argmax(dim=-1) == labels[train_idx]).float().mean()
            loss_train.backward()
            optimizer.step()
            train_loss_record.append(loss_train.item())
            train_acc_record.append(acc_train.item())
            print(f'Epoch [{epoch+1}/{args.epoch}]: Train loss: {loss_train.item():.4f}, acc: {acc_train:.4f}')

            model.eval()
            with torch.no_grad():
                output = model(features, adj)
                loss_valid = criterion(output[valid_idx], labels[valid_idx])
                acc_valid = (output[valid_idx].argmax(dim=-1) == labels[valid_idx]).float().mean()
                valid_loss_record.append(loss_valid.item())
                valid_acc_record.append(acc_valid.item())
            print(f'Epoch [{epoch+1}/{args.epoch}]: Valid loss: {loss_valid:.4f}, acc: {acc_valid:.4f}')

            if acc_valid > best_acc:
                best_acc = acc_valid
                checkpoint_path = args.saved_dir / 'model.pth'
                save_checkpoint(checkpoint_path, model, optimizer, epoch, best_acc)
                print('Saving model with acc {:.3f}...'.format(best_acc))
                early_stop_count = 0
            else:
                early_stop_count += 1
            
            if early_stop_count >= args.early_stop:
                print('\nModel is not improving, so we halt the training session.')
                break

            save_figure(str(args.saved_dir/(args.data + '_' + "train_valid_loss.png")), args.data + '_' + "train_valid_loss", train_loss_record, valid_loss_record)
            save_figure(str(args.saved_dir/(args.data + '_' + "train_valid_acc.png")), args.data + '_' + "train_valid_acc", train_acc_record, valid_acc_record)
    
    except KeyboardInterrupt:
        logger.info('Catch a KeyboardInterupt')

# %%
### 预测过程
def predict():
    checkpoint = torch.load(args.saved_dir/'model.pth')
    model.load_state_dict(checkpoint['modelstate'])
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = criterion(output[test_idx], labels[test_idx])
        acc_test = (output[test_idx].argmax(dim=-1) == labels[test_idx]).float().mean()
    print(f'Test loss: {loss_test.item()}')
    print(f'Test acc: {acc_test.item()}')

# %%
### 训练和预测
train()
predict()
print(f'Saved Path: {args.saved_dir}')
