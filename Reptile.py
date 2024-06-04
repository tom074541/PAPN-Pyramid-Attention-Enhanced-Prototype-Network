#!/usr/bin/env python3

import random
import copy
import numpy as np
import torch
import learn2learn as l2l
import random
import numpy as np
import torch
import learn2learn as l2l
from store_1 import LSTM
from store_1 import CNN
from torch import nn, optim
import pandas as pd
from sklearn.model_selection import train_test_split
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

def fast_adapt(batch, learner, adapt_opt, loss, adaptation_steps, shots,query_num, ways, batch_size, device):
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
        idx = torch.randint(
            adaptation_data.size(0),
            size=(batch_size, )
        )
        adapt_X = adaptation_data[idx]
        adapt_y = adaptation_labels[idx]
        adapt_opt.zero_grad()
        error = loss(learner(adapt_X), adapt_y)
        error.backward()
        adapt_opt.step()

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_error /= len(evaluation_data)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def main(
        ways=2,
        train_shots=5,
        test_shots=5,
        query_num=5,
        meta_lr=1.0,
        meta_lr_final=1.0,
        meta_bsz=5,
        fast_lr=0.001,
        train_bsz=10,
        test_bsz=15,
        train_steps=8,
        test_steps=50,
        iterations=1001,
        test_interval=100,
        save='',
        cuda=1,
        seed=38,
):
    cuda = bool(cuda)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    train_tasks, valid_tasks, test_tasks = l2l.vision.benchmarks.get_tasksets(
        'mini-imagenet',
        train_samples=2*train_shots,
        train_ways=ways,
        test_samples=2*test_shots,
        test_ways=ways,
        root='~/data',
    )

    # load the dataset
    train_data_path = r'Path to your data\ECG200_train.csv'  
    test_data_path = r'Path to your data\ECG200_test.csv'  
    train_data = pd.read_csv(train_data_path,header=None)
    test_data = pd.read_csv(test_data_path,header=None)

    # Seperate the data and the label, assume label is in the last column
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    X = test_data.iloc[:, :-1].values
    y = test_data.iloc[:, -1].values

    # 50% test, 50% validation
    X_validate, X_test, y_validate, y_test = train_test_split(X, y, test_size=0.5, random_state=41,stratify=y)

    X_train = np.expand_dims(X_train, axis=1)  
    X_validate = np.expand_dims(X_validate, axis=1)  
    X_test = np.expand_dims(X_test, axis=1)  

    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    X_validate, y_validate = torch.tensor(X_validate, dtype=torch.float32), torch.tensor(y_validate, dtype=torch.long)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_validate, y_validate)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_transforms = [
        NWays(train_dataset, ways),
        KShots(train_dataset, query_num + train_shots),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
    ]
    train_tasks = l2l.data.Taskset(train_dataset, task_transforms=train_transforms,num_tasks=10000)
    train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=True)

    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    valid_transforms = [
        NWays(valid_dataset, ways),
        KShots(valid_dataset, query_num + test_shots),
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
        KShots(test_dataset,query_num + test_shots),
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
    model = CNN()
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), meta_lr)
    adapt_opt = torch.optim.Adam(model.parameters(), lr=fast_lr, betas=(0, 0.999))
    adapt_opt_state = adapt_opt.state_dict()
    loss = torch.nn.CrossEntropyLoss(reduction='mean')

  #  train_inner_errors = []
  #  train_inner_accuracies = []
  #  valid_inner_errors = []
  #  valid_inner_accuracies = []
   # test_inner_errors = []
   # test_inner_accuracies = []
    count=0
    patience=10 
    best_accuracy=0
    avg_accuracy=0
    for iteration in range(iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0


        # anneal meta-lr
        frac_done = float(iteration) / iterations
        new_lr = frac_done * meta_lr_final + (1 - frac_done) * meta_lr
        for pg in opt.param_groups:
            pg['lr'] = new_lr

        # zero-grad the parameters
        for p in model.parameters():
            p.grad = torch.zeros_like(p.data)
        iterator_train = iter(train_loader)
        iterator_valid= iter(valid_loader)
        for task in range(meta_bsz):
            # Compute meta-training loss
            learner = copy.deepcopy(model)
            adapt_opt = torch.optim.Adam(
                learner.parameters(),
                lr=fast_lr,
                betas=(0, 0.999)
            )
            adapt_opt.load_state_dict(adapt_opt_state)
            batch = next(iterator_train)
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               adapt_opt,
                                                               loss,
                                                               train_steps,
                                                               train_shots,
                                                               query_num,
                                                               ways,
                                                               train_bsz,
                                                               device)
            adapt_opt_state = adapt_opt.state_dict()
            for p, l in zip(model.parameters(), learner.parameters()):
                p.grad.data.add_(l.data, alpha=-1.0)


            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()
            if iteration % test_interval == 0:
                # Compute meta-validation loss every 100 iterations
                learner = copy.deepcopy(model)
                adapt_opt = torch.optim.Adam(
                    learner.parameters(),
                    lr=fast_lr,
                    betas=(0, 0.999)
                )
                adapt_opt.load_state_dict(adapt_opt_state)
                batch = next(iterator_valid)
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   adapt_opt,
                                                                   loss,
                                                                   test_steps,
                                                                   test_shots,
                                                                   query_num,
                                                                   ways,
                                                                   test_bsz,
                                                                   device)
                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()
        avg_accuracy= meta_valid_accuracy / meta_bsz
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
        # Average the accumulated gradients and optimize
        for p in model.parameters():
            p.grad.data.mul_(1.0 / meta_bsz).add_(p.data)
        opt.step()
        # Print some metrics
        if iteration % test_interval == 0:
            print('\n')
            print('Iteration', iteration)
            print('Meta Train Error', meta_train_error / meta_bsz)
            print('Meta Train Accuracy', meta_train_accuracy / meta_bsz)
        if iteration % test_interval == 0:
            print('Meta Valid Error', meta_valid_error / meta_bsz)
            print('Meta Valid Accuracy', meta_valid_accuracy / meta_bsz)

    # Compute meta-testing loss
    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    iterator_test= iter(test_loader)
    for task in range(meta_bsz):
                learner = copy.deepcopy(model)
                adapt_opt = torch.optim.Adam(
                    learner.parameters(),
                    lr=fast_lr,
                    betas=(0, 0.999)
                )
                adapt_opt.load_state_dict(adapt_opt_state)
                batch = next(iterator_test)
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   adapt_opt,
                                                                   loss,
                                                                   test_steps,
                                                                   test_shots,
                                                                   query_num,
                                                                   ways,
                                                                   test_bsz,
                                                                   device)
                meta_test_error += evaluation_error.item()
                meta_test_accuracy += evaluation_accuracy.item()
    print('Meta Test Error', meta_test_error / meta_bsz)
    print('Meta Test Accuracy', meta_test_accuracy / meta_bsz)


###
        # Track quantities
        #train_inner_errors.append(meta_train_error / meta_bsz)
       # train_inner_accuracies.append(meta_train_accuracy / meta_bsz)
      #  if iteration % test_interval == 0:
         #   valid_inner_errors.append(meta_valid_error / meta_bsz)
         #   valid_inner_accuracies.append(meta_valid_accuracy / meta_bsz)
         #   test_inner_errors.append(meta_test_error / meta_bsz)
          #  test_inner_accuracies.append(meta_test_accuracy / meta_bsz)
###




if __name__ == '__main__':
    main()
