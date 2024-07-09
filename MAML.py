#!/usr/bin/env python3

import random
import numpy as np
import torch
import learn2learn as l2l
from store_1 import LSTM
from store_1 import CNN
from torch import nn, optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def fast_adapt(batch, learner, loss, adaptation_steps, shots,query_num, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Sort data samples by labels
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shots + query_num)
    for offset in range(shots):
        adaptation_indices[selection + offset] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def main(
        ways=2,
        shots=4,
        query_num=5,
        meta_lr=0.01,
        fast_lr=0.01,
        meta_batch_size=8,
        adaptation_steps=5,
        num_iterations=300,
        cuda=False,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    # Load the data
    train_data_path = r'Path to your data\ECG200_train.csv'  
    test_data_path = r'Path to your data\ECG200_train.csv'
    train_data = pd.read_csv(train_data_path,header=None)
    test_data = pd.read_csv(test_data_path,header=None)

    # Seperate data and label, assume label is in the laast column
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    X = test_data.iloc[:, :-1].values
    y = test_data.iloc[:, -1].values

    # Split into train and test sets if necessary
    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.7, random_state=41,stratify=y)

    # Split into test and validation sets
    X_validate, X_test, y_validate, y_test = train_test_split(X, y, test_size=0.5, random_state=41,stratify=y)

    # # Standardize
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_validate = scaler.transform(X_validate)
    # X_test = scaler.transform(X_test)

    X_train = np.expand_dims(X_train, axis=1) 
    X_validate = np.expand_dims(X_validate, axis=1)  
    X_test = np.expand_dims(X_test, axis=1)  

    # Transfer into Pytorch tensors
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    X_validate, y_validate = torch.tensor(X_validate, dtype=torch.float32), torch.tensor(y_validate, dtype=torch.long)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

    # reate TensorDataset
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_validate, y_validate)
    test_dataset = TensorDataset(X_test, y_test)

    # # Load the data
    # Data_path=r'E:\NUS2\Intern\learn2learn-master\Test_Data\feature_time_48k_2048_load_1.csv'
    # Data=pd.read_csv(Data_path)

    # #label_encoder = LabelEncoder()
    # #Data['fault'] = label_encoder.fit_transform(Data['fault'])

    # # Seperate data and label, assume label is in the laast column
    # X = Data.iloc[:, :-1].values
    # y = Data.iloc[:, -1].values
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    # # 1550 train, 375 test, 375 validation
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=750, random_state=41,stratify=y)
    # X_test, X_validate, y_test, y_validate = train_test_split(X_test, y_test, test_size=0.5, random_state=41,stratify=y_test)

    # # Add channel
    # X_train = np.expand_dims(X_train, axis=1)  
    # X_validate = np.expand_dims(X_validate, axis=1)  
    # X_test = np.expand_dims(X_test, axis=1)  

    # # To PyTorch tensor
    # X_train = torch.tensor(X_train, dtype=torch.float32)
    # X_validate = torch.tensor(X_validate, dtype=torch.float32)
    # X_test = torch.tensor(X_test, dtype=torch.float32)

    # label_encoder = LabelEncoder()
    # y_train = label_encoder.fit_transform(y_train)
    # y_train = torch.tensor(y_train, dtype=torch.int64)
   
    # y_test = label_encoder.fit_transform(y_test)
    # y_test = torch.tensor(y_test, dtype=torch.int64)
  
    # y_validate = label_encoder.fit_transform(y_validate)
    # y_validate = torch.tensor(y_validate, dtype=torch.int64)

    # # To TensorDataset
    # train_dataset = TensorDataset(X_train, y_train)
    # valid_dataset = TensorDataset(X_validate, y_validate)
    # test_dataset = TensorDataset(X_test, y_test)
    
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_transforms = [
        NWays(train_dataset, ways),
        KShots(train_dataset, query_num + shots),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
    ]
    train_tasks = l2l.data.Taskset(train_dataset, task_transforms=train_transforms,num_tasks=10000)
    train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=True)

    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    valid_transforms = [
        NWays(valid_dataset, ways),
        KShots(valid_dataset, query_num + shots),
        LoadData(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.Taskset(
        valid_dataset,
        task_transforms=valid_transforms,
        num_tasks=10000,
    )
    valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=True)

    test_dataset = l2l.data.MetaDataset(test_dataset)
    test_transforms = [
        NWays(test_dataset, ways),
        KShots(test_dataset,query_num + shots),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
    ]
    test_tasks = l2l.data.Taskset(
        test_dataset,
        task_transforms=test_transforms,
        num_tasks=10000,
    )
    test_loader = DataLoader(test_tasks, pin_memory=True, shuffle=True)
    # Create model
    input_size=1
    hidden_size=32
    num_layers=2
    num_classes=2
    model =CNN()
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    count=0
    patience=10 
    best_accuracy=0
    avg_accuracy=0
    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        iterator_train = iter(train_loader)
        iterator_valid = iter(valid_loader)
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = next(iterator_train)
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               query_num,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = next(iterator_valid)
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               query_num,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        avg_accuracy= meta_valid_accuracy / meta_batch_size
        # Check for early stopping
        if iteration % 100 == 0:
            if best_accuracy < avg_accuracy:
                best_accuracy = avg_accuracy
                count = 0
            else:
                count += 1
                if count >= patience:
                    print("Validation performance did not improve for 1000 epochs. Stopping training.")
                    break


        if iteration % 5 == 0:
            print('\n')
            print('Iteration', iteration)
            print('Meta Train Error', meta_train_error / meta_batch_size)
            print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
            print('Meta Valid Error', meta_valid_error / meta_batch_size)
            print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)
        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    iterator_test= iter(test_loader)
    for task in range(meta_batch_size):
        # Compute meta-testing loss
        learner = maml.clone()
        batch = next(iterator_test)
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           learner,
                                                           loss,
                                                           adaptation_steps,
                                                           shots,
                                                           query_num,
                                                           ways,
                                                           device)
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()
    print('Meta Test Error', meta_test_error / meta_batch_size)
    print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)


if __name__ == '__main__':
    main()
