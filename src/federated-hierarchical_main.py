#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details


# BUILD MODEL
def build_model(args, train_dataset):
    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')
        
    return global_model


# Defining the training function
def fl_train(args, train_dataset, cluster_global_model, cluster, usergrp, epochs):
    
    cluster_train_loss, cluster_train_accuracy = [], []
    cluster_val_acc_list, cluster_net_list = [], []
    cluster_cv_loss, cluster_cv_acc = [], []
    # print_every = 1
    cluster_val_loss_pre, counter = 0, 0

    for epoch in range(epochs):
        cluster_local_weights, cluster_local_losses = [], []
        # print(f'\n | Cluster Training Round : {epoch+1} |\n')

        cluster_global_model.train()
        m = max(int(args.frac * len(cluster)), 1)
        idxs_users = np.random.choice(cluster, m, replace=False)


        for idx in idxs_users:
            cluster_local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=usergrp[idx], logger=logger)
            cluster_w, cluster_loss = cluster_local_model.update_weights(model=copy.deepcopy(cluster_global_model), global_round=epoch)
            cluster_local_weights.append(copy.deepcopy(cluster_w))
            cluster_local_losses.append(copy.deepcopy(cluster_loss))
            # print('| Global Round : {} | User : {} | \tLoss: {:.6f}'.format(epoch, idx, cluster_loss))

        # averaging global weights
        cluster_global_weights = average_weights(cluster_local_weights)

        # update global weights
        cluster_global_model.load_state_dict(cluster_global_weights)

        cluster_loss_avg = sum(cluster_local_losses) / len(cluster_local_losses)
        cluster_train_loss.append(cluster_loss_avg)

    return cluster_global_weights, cluster_loss_avg
    




if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # ======= Splitting into clusters. FL groups ======= 
    cluster_size = args.num_users / args.num_clusters
    print("Each cluster size: ", cluster_size)

    # Cluster 1
    A1 = np.arange(cluster_size, dtype=int)
    user_groupsA = {k:user_groups[k] for k in A1 if k in user_groups}
    print("Size of cluster 1: ", len(user_groupsA))
    # Cluster 2
    B1 = np.arange(cluster_size, cluster_size+cluster_size, dtype=int)
    user_groupsB = {k:user_groups[k] for k in B1 if k in user_groups}
    print("Size of cluster 2: ", len(user_groupsB))
    # Cluster 3
    C1 = np.arange(2*cluster_size, 3*cluster_size, dtype=int)
    user_groupsC = {k:user_groups[k] for k in C1 if k in user_groups}
    print("Size of cluster 3: ", len(user_groupsC))
    # Cluster 4
    D1 = np.arange(3*cluster_size, 4*cluster_size, dtype=int)
    user_groupsD = {k:user_groups[k] for k in D1 if k in user_groups}
    print("Size of cluster 4: ", len(user_groupsD))

    # MODEL PARAM SUMMARY
    global_model = build_model(args, train_dataset)
    pytorch_total_params = sum(p.numel() for p in global_model.parameters())
    print("Model total number of parameters: ", pytorch_total_params)

    # from torchsummary import summary
    # summary(global_model, (1, 28, 28))
    # global_model.parameters()

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()


    # ======= Set the cluster models to train and send it to device. =======
    # Cluster A
    cluster_modelA = build_model(args, train_dataset)
    cluster_modelA.to(device)
    cluster_modelA.train()
    # copy weights
    cluster_modelA_weights = cluster_modelA.state_dict()
    # Cluster B
    cluster_modelB = build_model(args, train_dataset)
    cluster_modelB.to(device)
    cluster_modelB.train()
    # copy weights
    cluster_modelB_weights = cluster_modelB.state_dict()
    # Cluster C
    cluster_modelC = build_model(args, train_dataset)
    cluster_modelC.to(device)
    cluster_modelC.train()
    # copy weights
    cluster_modelC_weights = cluster_modelC.state_dict()
    # Cluster D
    cluster_modelD = build_model(args, train_dataset)
    cluster_modelD.to(device)
    cluster_modelD.train()
    # copy weights
    cluster_modelD_weights = cluster_modelD.state_dict()


    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0
    testacc_check, epoch = 0, 0

    # for epoch in tqdm(range(args.epochs)):
    while testacc_check < args.test_acc:
        local_weights, local_losses, local_accuracies= [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        
        # ============== TRAIN ==============
        global_model.train()
        
        # Cluster A
        A_weights, A_losses = fl_train(args, train_dataset, cluster_modelA, A1, user_groupsA, args.Cepochs)
        local_weights.append(copy.deepcopy(A_weights))
        local_losses.append(copy.deepcopy(A_losses))        
        # Cluster B
        B_weights, B_losses = fl_train(args, train_dataset, cluster_modelB, B1, user_groupsB, args.Cepochs)
        local_weights.append(copy.deepcopy(B_weights))
        local_losses.append(copy.deepcopy(B_losses))
        # Cluster C
        C_weights, C_losses = fl_train(args, train_dataset, cluster_modelC, C1, user_groupsC, args.Cepochs)
        local_weights.append(copy.deepcopy(C_weights))
        local_losses.append(copy.deepcopy(C_losses))        
        # Cluster D
        D_weights, D_losses = fl_train(args, train_dataset, cluster_modelD, D1, user_groupsD, args.Cepochs)
        local_weights.append(copy.deepcopy(D_weights))
        local_losses.append(copy.deepcopy(D_losses))
        
        
        # averaging global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        
        # ============== EVAL ============== 
        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))
        # Add
        testacc_check = 100*train_accuracy[-1]
        epoch = epoch + 1

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    # print(f' \n Results after {args.epochs} global rounds of training:')
    print(f"\nAvg Training Stats after {epoch} global rounds:")
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    format(args.dataset, args.model, epoch, args.frac, args.iid,
           args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)