#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils import get_dataset
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
import pickle
import time

if __name__ == '__main__':
    args = args_parser()
    if args.gpu_id:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    start_time = time.time()

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)
    with torch.no_grad():
        torch.cuda.empty_cache()

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
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

    # Set the model to train and send it to device.
    global_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=False)
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    trainloader = DataLoader(train_dataset, batch_size=5096, shuffle=True, pin_memory=True, num_workers=1)
    criterion = torch.nn.NLLLoss().to(device)
    epoch_loss = []
    loss = None
    for epoch in range(args.epochs):    
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


    # testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))


    # Saving the objects train_loss, test_acc, test_loss:
    file_name = '../save/objects/BaseSGD_{}_{}_epoch[{}]_lr[{}]_iid[{}].pkl'.\
        format(args.dataset, args.model, epoch, args.lr, args.iid)

    with open(file_name, 'wb') as f:
        pickle.dump([epoch_loss, test_acc, test_loss], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))


    # # Plot loss
    # plt.figure()
    # plt.plot(range(len(epoch_loss)), epoch_loss)
    # plt.xlabel('epochs')
    # plt.ylabel('Train loss')
    # plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
    #                                              args.epochs))


