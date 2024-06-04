#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
from sklearn.preprocessing import StandardScaler, LabelEncoder
from store import CNNAttention
from store import VGG16
from store import Resnet18
from store import Resnet18PAttention
from store import LSTMAttention
from store import LSTM
from store import CNN
from store import CNNAttentionGate
def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(model, batch, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)
    n_items = shot * ways

    # Sort data samples by labels
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()
    logits = pairwise_distances_logits(query, support)
    loss = F.cross_entropy(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--shot', type=int, default=6)
    parser.add_argument('--test-way', type=int, default=10)
    parser.add_argument('--test-shot', type=int, default=6)
    parser.add_argument('--test-query', type=int, default=2)
    parser.add_argument('--train-query', type=int, default=2)
    parser.add_argument('--train-way', type=int, default=10)
    parser.add_argument('--gpu', default=1)
    args = parser.parse_args()
    print(args)

    device = torch.device('cpu')
    if args.gpu and torch.cuda.device_count():
        print("Using gpu")
        torch.cuda.manual_seed(41)
        device = torch.device('cuda')
    input_size=1
    hidden_size=16
    num_layers=2
    num_heads=8
    d_model=32 #channels
    #model=Resnet18Attention(d_model,num_heads)
    #model=Resnet18PAttention(d_model,num_heads)
    #model=CNNAttention(d_model,num_heads)
    #model = VGG16()
    #model=Resnet18()
    model=CNN()
    model.to(device)

    # Load the data
    Data_path=r'D:\Code\learn2learn-master\Test_Data\feature_time_48k_2048_load_1.csv'
    Data=pd.read_csv(Data_path)

    #label_encoder = LabelEncoder()
    #Data['fault'] = label_encoder.fit_transform(Data['fault'])

    # Seperate data and label, assume label is in the laast column
    X = Data.iloc[:, :-1].values
    y = Data.iloc[:, -1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 1550 train, 375 test, 375 validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=750, random_state=41,stratify=y)
    X_test, X_validate, y_test, y_validate = train_test_split(X_test, y_test, test_size=0.5, random_state=41,stratify=y_test)

    # Add channel
    X_train = np.expand_dims(X_train, axis=1)  
    X_validate = np.expand_dims(X_validate, axis=1)  
    X_test = np.expand_dims(X_test, axis=1)  

    # To PyTorch tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_validate = torch.tensor(X_validate, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_train = torch.tensor(y_train, dtype=torch.int64)
   
    y_test = label_encoder.fit_transform(y_test)
    y_test = torch.tensor(y_test, dtype=torch.int64)
  
    y_validate = label_encoder.fit_transform(y_validate)
    y_validate = torch.tensor(y_validate, dtype=torch.int64)

    # To TensorDataset
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_validate, y_validate)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_transforms = [
        NWays(train_dataset, args.train_way),
        KShots(train_dataset, args.train_query + args.shot),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
    ]
    train_tasks = l2l.data.Taskset(train_dataset, task_transforms=train_transforms,num_tasks=2000)
    train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=True)

    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    valid_transforms = [
        NWays(valid_dataset, args.test_way),
        KShots(valid_dataset, args.test_query + args.test_shot),
        LoadData(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.Taskset(
        valid_dataset,
        task_transforms=valid_transforms,
        num_tasks=100,
    )
    valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=True)

    test_dataset = l2l.data.MetaDataset(test_dataset)
    test_transforms = [
        NWays(test_dataset, args.test_way),
        KShots(test_dataset, args.test_query + args.test_shot),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
    ]
    test_tasks = l2l.data.Taskset(
        test_dataset,
        task_transforms=test_transforms,
        num_tasks=100,
    )
    test_loader = DataLoader(test_tasks, pin_memory=True, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5)
    count=0
    patience=10
    best_accuracy=0
    for epoch in range(1, args.max_epoch + 1):
        model.train()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        # Create iterator
        iterator = iter(train_loader)
        for i in range(100):
            batch = next(iterator)
        
            loss, acc = fast_adapt(model,
                                batch,
                                args.train_way,
                                args.shot,
                                args.train_query,
                                metric=pairwise_distances_logits,
                                device=device)

          
            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        print('epoch {}, train, loss={:.4f} acc={:.4f}'.format(
            epoch, n_loss/loss_ctr, n_acc/loss_ctr))

        model.eval()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        avg_loss=0

        for i, batch in enumerate(valid_loader):
            loss, acc = fast_adapt(model,
                                   batch,
                                   args.test_way,
                                   args.test_shot,
                                   args.test_query,
                                   metric=pairwise_distances_logits,
                                   device=device)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc
            avg_loss=n_loss/loss_ctr
            avg_accuracy=n_acc/loss_ctr
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(
                epoch, avg_loss, avg_accuracy))

        # Check for early stopping
        if best_accuracy < avg_accuracy:
            best_accuracy = avg_accuracy
            count = 0
        else:
            count += 1
            if count >= patience:
                print("Validation performance did not improve for {} epochs. Stopping training.".format(patience))
                break
        if count >= patience:
            break

    model.eval()
    loss_ctr = 0
    n_acc = 0

    for i, batch in enumerate(test_loader, 1):
        loss, acc = fast_adapt(model,
                               batch,
                               args.test_way,
                               args.test_shot,
                               args.test_query,
                               metric=pairwise_distances_logits,
                               device=device)
        loss_ctr += 1
        n_acc += acc
        print('batch {}: {:.2f}({:.2f})'.format(
            i, n_acc/loss_ctr * 100, acc * 100))
        
