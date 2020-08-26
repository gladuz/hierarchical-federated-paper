#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, set_device, build_model, fl_train
import math
import random


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # Select CPU or GPU
    device = set_device(args)

    # load dataset and user groups
    train_dataset, test_dataset, user_groupsold = get_dataset(args)

    # user_groups = user_groupsold
    # keylist = list(user_groups.keys())
    # ======= Shuffle dataset ======= 
    keys =  list(user_groupsold.keys())
    #random.shuffle(keys)
    user_groups = dict()
    for key in keys:
        user_groups.update({key:user_groupsold[key]})
    # print(user_groups.keys()) 
    keylist = list(user_groups.keys())
    print("keylist: ", keylist)
    # ======= Splitting into clusters. FL groups ======= 
    cluster_size = int(args.num_users / args.num_clusters)    
    # cluster_size = 50
    print("Each cluster size: ", cluster_size)

    cluster_users = []
    cluster_user_groups = []
    for i in range(args.num_clusters):
        cluster_users.append(keylist[i*cluster_size:(i+1)*cluster_size])
        user_groups_local = {k:user_groups[k] for k in cluster_users[i] if k in user_groups}
        cluster_user_groups.append(user_groups)

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
    cluster_models = []
    for i in range(args.num_clusters):
        model = build_model(args, train_dataset)
        model.to(device)
        model.train()
        cluster_models.append(model)


    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0
    testacc_check, epoch = 0, 0 
    idx = np.random.randint(0,99)

    # for epoch in tqdm(range(args.epochs)):
    for epoch in range(args.epochs):
    # while testacc_check < args.test_acc or epoch < args.epochs:
    # while epoch < args.epochs:        
        local_weights, local_losses, local_accuracies= [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        
        # ============== TRAIN ==============
        global_model.train()
        
        # ======= Dynamic training cluster ======= 
        for i in range(args.num_clusters):
            cluster_model = cluster_models[i]
            losses, weights = None, None
            for i in range(3):
                _, weights, losses = fl_train(args, train_dataset, cluster_model, cluster_users[i], cluster_user_groups[i], args.Cepochs, logger)  
                cluster_model.load_state_dict(weights)
            local_weights.append(copy.deepcopy(weights))
            local_losses.append(copy.deepcopy(losses))    
            cluster_models[i] = global_model


        # # ===== Cluster A ===== 
        # _, A_weights, A_losses = fl_train(args, train_dataset, cluster_modelA, A1, user_groupsA, args.Cepochs, logger)        
        # local_weights.append(copy.deepcopy(A_weights))
        # local_losses.append(copy.deepcopy(A_losses))    
        # cluster_modelA = global_model #= A_model        
        # # ===== Cluster B ===== 
        # B_model, B_weights, B_losses = fl_train(args, train_dataset, cluster_modelB, B1, user_groupsB, args.Cepochs, logger)
        # local_weights.append(copy.deepcopy(B_weights))
        # local_losses.append(copy.deepcopy(B_losses))
        # cluster_modelB = global_model #= B_model 
        # # ===== Cluster C ===== 
        # C_model, C_weights, C_losses = fl_train(args, train_dataset, cluster_modelC, C1, user_groupsC, args.Cepochs, logger)
        # local_weights.append(copy.deepcopy(C_weights))
        # local_losses.append(copy.deepcopy(C_losses))   
        # cluster_modelC = global_model #= C_model      
        # # ===== Cluster D ===== 
        # D_model, D_weights, D_losses = fl_train(args, train_dataset, cluster_modelD, D1, user_groupsD, args.Cepochs, logger)
        # local_weights.append(copy.deepcopy(D_weights))
        # local_losses.append(copy.deepcopy(D_losses))
        # cluster_modelD= global_model #= D_model 
        
        
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
        # print("========== idx ========== ", idx)
        for c in range(args.num_users):
        # for c in range(cluster_size):
        # C = np.random.choice(keylist, int(args.frac * args.num_users), replace=False) # random set of clients
        # print("C: ", C)
        # for c in C:
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
    file_name = '../save/objects/old_HFL4_{}_{}_{}_lr[{}]_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    format(args.dataset, args.model, epoch, args.lr, args.frac, args.iid,
           args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)
    
    # PLOTTING (optional)
    matplotlib.use('Agg')

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/old_fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))