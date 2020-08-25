import matplotlib
import matplotlib.pyplot as plt
import pickle

with open('../save/objects/old_HFL4_mnist_cnn_5_lr[0.01]_C[0.3]_iid[0]_E[1]_B[10].pkl', 'rb') as f, open('../save/objects/HFL4_mnist_cnn_5_lr[0.01]_C[0.3]_iid[0]_E[1]_B[10].pkl', 'rb') as new_f:
    loss, acc = pickle.load(new_f)
    old_loss, old_acc = pickle.load(f)
    plt.figure(0)
    plt.title('Average Loss vs Communication rounds')
    plt.plot(range(len(old_loss)), old_loss, color='b', label='Old way')
    plt.plot(range(len(loss)), loss, color='r', label='New way')
    plt.ylabel('Average Loss')
    plt.xlabel('Communication Rounds')
    plt.legend()
    plt.savefig('../save/fed_paper_loss.png')

    plt.figure(1)
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(old_acc)), old_acc, color='b', label='Old way')
    plt.plot(range(len(acc)), acc, color='r', label='New way')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.legend()
    plt.savefig('../save/fed_paper_acc.png')

