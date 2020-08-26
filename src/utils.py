#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from sys import exit
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, mnist_noniid_clustered
from sampling import cifar_iid, cifar_noniid
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
import update
#from update import LocalUpdate, test_inference



def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       # transform=apply_transform)
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)
        
        # test_dataset = datasets.MNIST(data_dir, train=False, download=True,
        #                               transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            print("Dataset: CIFAR10 IID")
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                print("Dataset: CIFAR10 equal Non-IID")
                user_groups = cifar_noniid(train_dataset, args.num_users)


    elif args.dataset == 'mnist': 
        data_dir = '../data/mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            print("Dataset: MNIST IID")
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                print("Dataset: MNIST unequal Non-IID")
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose equal splits for every user
                print("Dataset: MNIST equal Non-IID")
                if args.clustered_data:
                    user_groups = mnist_noniid_clustered(train_dataset, args.num_users)
                else:
                    user_groups = mnist_noniid(train_dataset, args.num_users)
                
    else:
        exit("No such dataset: " + args.dataset)
        
    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def set_device(args):
    # Select CPU or GPU
    if not args.gpu or not torch.cuda.is_available():
        device=torch.device('cpu')
    else:
        # Check that GPU is indeed available
        device = torch.device(args.gpu_id)
    
    return device
    
    
def build_model(args, train_dataset):
    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        model = MLP(dim_in=len_in, dim_hidden=args.mlpdim,
                               dim_out=args.num_classes)
    else:
        exit('Error- unrecognized model: ' + args.model)
        
    return model
    
    
def fl_train(args, train_dataset, cluster_global_model, cluster, usergrp, epochs, logger, cluster_dtype=torch.float32):
    """
    Defining the training function.
    """
    
    cluster_train_loss, cluster_train_acc = [], []
    cluster_val_acc_list, cluster_net_list = [], []
    cluster_cv_loss, cluster_cv_acc = [], []
    # print_every = 1
    cluster_val_loss_pre, counter = 0, 0

    for epoch in range(epochs):
        cluster_local_weights, cluster_local_losses = [], []
        # print(f'\n | Cluster Training Round : {epoch+1} |\n')

        cluster_global_model.train()
        # m = max(int(args.frac * len(cluster)), 1)
        # m = max(int(math.ceil(args.frac * len(cluster))), 1)
        m = min(int(len(cluster)), 10)
        # print("=== m ==== ", m)
        # m = 10
        idxs_users = np.random.choice(cluster, m, replace=False)


        for idx in idxs_users:
            cluster_local_model = update.LocalUpdate(args=args, dataset=train_dataset, idxs=usergrp[idx], logger=logger)
            cluster_w, cluster_loss = cluster_local_model.update_weights(model=copy.deepcopy(cluster_global_model), global_round=epoch, dtype=cluster_dtype)
            cluster_local_weights.append(copy.deepcopy(cluster_w))
            cluster_local_losses.append(copy.deepcopy(cluster_loss))
            # print('| Global Round : {} | User : {} | \tLoss: {:.6f}'.format(epoch, idx, cluster_loss))

        # averaging global weights
        cluster_global_weights = average_weights(cluster_local_weights)

        # update global weights
        cluster_global_model.load_state_dict(cluster_global_weights)

        cluster_loss_avg = sum(cluster_local_losses) / len(cluster_local_losses)
        cluster_train_loss.append(cluster_loss_avg)

        # ============== EVAL ============== 
        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        cluster_global_model.eval()
        # C = np.random.choice(cluster, m, replace=False) # random set of clients
        # print("C: ", C)
        # for c in C:
        # for c in range(len(cluster)):  
        for c in idxs_users:      
            cluster_local_model = update.LocalUpdate(args=args, dataset=train_dataset, idxs=usergrp[c], logger=logger)
            # local_model = LocalUpdate(args=args, dataset=train_dataset,idxs=user_groups[idx], logger=logger)
            acc, loss = cluster_local_model.inference(model=cluster_global_model, dtype=cluster_dtype)
            list_acc.append(acc)
            list_loss.append(loss)
        # cluster_train_acc.append(sum(list_acc)/len(list_acc))
        # Add
    # print("Cluster accuracy: ", 100*cluster_train_acc[-1]) 
    print("Cluster accuracy: ", 100*sum(list_acc)/len(list_acc)) 

    return cluster_global_model, cluster_global_weights, cluster_loss_avg

