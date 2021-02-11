import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('../save/objects/HFL4_mnist_cnn_50_lr[0.01]_C[0.1]_iid[0]_E[60]_B[10].pkl', 'rb') as f, open('../save/objects/HFL4_mnist_cnn_50_lr[0.007783125570686415]_C[0.2]_iid[0]_E[60]_B[10]_60iternonIID_ORIGINAL_.pkl', 'rb') as new_f\
    , open('../save/objects/HFL4_mnist_cnn_50_lr[0.007783125570686415]_C[0.1]_iid[0]_E[25]_B[10].pkl', 'rb') as edge_fl, open('../save/objects/FL_van_FL.pkl', 'rb') as van_fl:
    loss, acc = pickle.load(new_f)
    paper_loss, paper_acc = pickle.load(f)
    edge_loss, edge_acc = pickle.load(edge_fl)
    van_loss, van_acc = pickle.load(van_fl)
    # plt.figure(0)
    # plt.plot(range(len(paper_loss)), paper_loss, color='b-', label='Edge-aided HierFL')
    # plt.plot(range(len(loss)), loss, color='r--', label='Existing [6]')
    # plt.plot(range(len(edge_loss)), edge_loss, color='g:', label='Edge based FL')
    # plt.ylabel('Average Loss')
    # plt.xlabel('Global Communication Rounds')
    # plt.legend()
    # plt.savefig('../save/test_edge-hier-vanilla_loss.png')

    plt.figure(1)
    plt.plot(range(len(paper_acc)), paper_acc, 'b-', label='Proposed')
    plt.plot(range(len(van_acc)), van_acc, 'c-.', label='Existing [2]')
    plt.plot(range(len(acc)), acc, 'r--', label='Existing [9]')
    plt.plot(range(len(edge_acc)), edge_acc, 'g:', label='Existing [10]')

    plt.ylabel('Test Accuracy')
    plt.xlabel('Number of communication rounds')
    plt.legend()
    plt.savefig('../save/test1_edge-hier-vanilla_acc.eps', format='eps')
    print(van_acc[-1])


    plt.figure(2)
    plt.plot(range(len(paper_acc)), paper_acc, color='b', label='Edge-aided HierFL')
    plt.plot(range(len(acc)), acc, color='r', label='HierFAVG')
    plt.plot(range(len(edge_acc)), edge_acc, color='g', label='Edge based FL')
    plt.plot(range(len(van_acc)), van_acc, color='g', label='Edge based FL')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global Communication Rounds')
    plt.legend()
    #plt.savefig('../save/fed_paper_acc_edge_vanilla.png')

