import argparse
import logging
import sys
import datetime
import yaml
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import math
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader

### 数据处理
train_valid_data = pd.read_json("yelp_phoenix_academic_dataset/yelp_academic_dataset_review.json", lines=True)
test_data = pd.read_json("yelp_phoenix_academic_dataset/test.json", lines=True)
review_data = train_valid_data[~train_valid_data["review_id"].isin(test_data["review_id"])] #从训练集中删除在测试集中的数据

train_valid_data = train_valid_data[["text", "stars"]]
test_data = test_data[["text", "stars"]]

# 获取一个分词器
tokenizer = get_tokenizer("basic_english")

# 构建词汇表  
def build_vocabulary(datasets):
    for text in datasets:
            yield tokenizer(text) # 最大的token数目为1219
all_text = np.concatenate([train_valid_data["text"].values, test_data["text"].values])
vocab = build_vocab_from_iterator(build_vocabulary(all_text), min_freq=1, specials=["<UNK>"])

### 处理参数
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser('time series forecasting in the financial sector')

    parser.add_argument('--cpu', action='store_true', help='wheather to use cpu , if not set, use gpu')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--train_valid_ratio', type=float, default=0.2)
    parser.add_argument('--optim', type=str, default='adam', choices=['sgd','adam','adamw'])
    parser.add_argument('--loss_method', type=str, default='ce', choices=['mse', 'ce'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--lr_decay', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--best_loss', type=float, default=math.inf)

    parser.add_argument('--saved_dir', type=str, default='./saved_dir')

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

### 划分数据集、验证集
def train_valid_split(data, train_valid_ratio):
    valid_size = int(len(data) * train_valid_ratio)
    train_size = len(data) - valid_size

    return data.head(train_size), data.tail(valid_size)

### Dataset
class TextDataset(Dataset):
    def __init__(self, data, max_token=1000):
        self.features = data["text"]
        self.tokens = self.features.apply(lambda text : vocab(tokenizer(text)))
        self.tokens = self.tokens.apply(lambda token : token[:max_token]+[0]*max(0,max_token-len(token)))
        self.labels = data["stars"]

    def __getitem__(self, index):
        feature = torch.tensor(self.tokens.iloc[index])
        label = self.labels.iloc[index]-1
        return feature, label

    def __len__(self):
        return len(self.features)


### Net
class RNNClassifier(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=50)
        self.rnn = nn.RNN(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 5)
        self.device = device

    def forward(self, x):
        embeddings = self.embedding_layer(x)
        hidden = torch.zeros(1, x.size()[0], 50).to(self.device)
        output, _ = self.rnn(embeddings, hidden)
        return self.linear(output[:, -1])
    
### 保存损失和准确率
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False
def save_loss(output_dir, train_loss_record, valid_loss_record):
    plt.figure()
    plt.plot(np.arange(len(train_loss_record)), train_loss_record, label='train')
    plt.plot(np.arange(len(valid_loss_record)), valid_loss_record, label='valid')
    plt.legend()
    plt.grid(True)
    plt.title('train_valid_loss')
    plt.savefig(str(output_dir / 'train_valid_loss.png'))
    plt.close()
def save_acc(output_dir, train_acc_record, valid_acc_record):
    plt.figure()
    plt.plot(np.arange(len(train_acc_record)), train_acc_record, label='train')
    plt.plot(np.arange(len(valid_acc_record)), valid_acc_record, label='valid')
    plt.legend()
    plt.grid(True)
    plt.title('train_valid_acc')
    plt.savefig(str(output_dir / 'train_valid_acc.png'))
    plt.close()

### 构建模型
def _make_model(device):
    model = RNNClassifier(device).to(device)
    model_parameters = filter(lambda p:p.requires_grad,model.parameters())
    n_params = sum([p.numel() for p in model_parameters])
    logger.info('Model Setting ...')
    logger.info(f'Number of model params: {n_params}')
    return model

### 构建优化器
def _make_optimizer(args, model):
    logger.info(f'Using {args.optim} Optimizer ......')
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.epsilon)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, nesterov=True, weight_decay=args.weight_decay)
    return optimizer

### 构建损失
def _make_criterion(args):
    if args.loss_method == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_method == 'mse':
        criterion = nn.MSELoss(reduction='mean')
    return criterion

### 保存模型
def save_checkpoint(checkpoint_path, model, optimizer, epoch, best_loss):
    if isinstance(model,(torch.nn.parallel.DistributedDataParallel,torch.nn.DataParallel)):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    checkpoint = {
        'modelstate':model_state_dict,
        'optimstate':optimizer.state_dict(),
        'epoch_id':epoch,
        'beat_loss':best_loss
        }
    torch.save(checkpoint, checkpoint_path)

### 训练过程
def train(args):
    device = torch.device('cuda' if not args.cpu else 'cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.cpu:
        torch.cuda.manual_seed(args.seed)

    model = _make_model(device)
    optimizer = _make_optimizer(args, model)
    criterion = _make_criterion(args)

    train_data, valid_data = train_valid_split(train_valid_data, args.train_valid_ratio)
    train_dataset, valid_dataset = TextDataset(train_data), TextDataset(valid_data)
    logger.info(f'Number of train samples: {len(train_dataset)}')
    logger.info(f'Number of valid samples: {len(valid_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    best_loss = args.best_loss
    early_stop_count = 0

    train_loss_record = []
    valid_loss_record = []
    train_acc_record = []
    valid_acc_record = []

    try:
        for epoch in range(0, args.epoch):
            model.train()
            loss_record = []
            acc_record = []
            train_pbar = tqdm(train_loader, position=0, leave=True)
            for x, y in train_pbar:
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                pred = model(x)    
                loss = criterion(pred, y)
                loss.backward()           
                optimizer.step()
                l_ = loss.detach().item()
                loss_record.append(l_)
                acc = (pred.argmax(dim=-1) == y.to(device)).float().mean()
                acc_record.append(acc.detach().item())
                train_pbar.set_description(f'Epoch [{epoch+1}/{args.epoch}]')
                train_pbar.set_postfix({'loss': f'{l_:.5f}'})
            mean_train_loss = sum(loss_record)/len(loss_record)
            train_loss_record.append(mean_train_loss)
            mean_train_acc = sum(acc_record)/len(acc_record)
            train_acc_record.append(mean_train_acc)

            print(f'Epoch [{epoch+1}/{args.epoch}]: Train loss: {mean_train_loss:.4f}, acc: {mean_train_acc:.4f}')

            model.eval()
            loss_record = []
            acc_record = []
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    pred = model(x)
                    loss = criterion(pred, y)
                    acc = (pred.argmax(dim=-1) == y.to(device)).float().mean()
                loss_record.append(loss.item())
                acc_record.append(acc.detach().item())
            
            mean_valid_loss = sum(loss_record)/len(loss_record)
            valid_loss_record.append(mean_valid_loss)
            mean_valid_acc = sum(acc_record)/len(acc_record)
            valid_acc_record.append(mean_valid_acc)

            print(f'Epoch [{epoch+1}/{args.epoch}]: Valid loss: {mean_valid_loss:.4f}, acc: {mean_valid_acc:.4f}')

            if mean_valid_loss < best_loss:
                best_loss = mean_valid_loss
                checkpoint_path = args.saved_dir / 'model.pth'
                save_checkpoint(checkpoint_path, model, optimizer, epoch, best_loss)
                print('Saving model with loss {:.3f}...'.format(best_loss))
                early_stop_count = 0
            else:
                early_stop_count += 1
            
            if early_stop_count >= args.early_stop:
                print('\nModel is not improving, so we halt the training session.')
                break

            save_loss(args.saved_dir, train_loss_record, valid_loss_record)
            save_acc(args.saved_dir, train_acc_record, valid_acc_record)
    
    except KeyboardInterrupt:
        logger.info('Catch a KeyboardInterupt')

### 预测
def predict(args):
    device = torch.device('cuda' if not args.cpu else 'cpu')

    model = _make_model(device)
    checkpoint = torch.load(args.saved_dir/'model.pth')
    model.load_state_dict(checkpoint['modelstate'])

    criterion = _make_criterion(args)

    test_dataset = TextDataset(test_data)
    logger.info(f'Number of test samples: {len(test_dataset)}')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    loss_record = []
    acc_record = []

    model.eval()
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            loss = criterion(pred, y)
            acc = (pred.argmax(dim=-1) == y.to(device)).float().mean()
        loss_record.append(loss.item())
        acc_record.append(acc.detach().item())
    mean_test_loss = sum(loss_record)/len(loss_record)
    mean_test_acc = sum(acc_record)/len(acc_record)
    print(f'Test loss: {mean_test_loss:.4f}')
    print(f'Test acc: {mean_test_acc:.4f}')

args = get_args()
train(args)
predict(args)