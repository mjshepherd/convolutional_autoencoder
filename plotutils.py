import matplotlib.pyplot as plt
import numpy as np
import pdb


def ShowImages(imgs):
    """Show the cluster centers as images."""
    plt.figure(1)
    plt.clf()
    for i in range(0, 5):
        for j in range(len(imgs)):
            plt.subplot(5, 3, (j+1) + (i*3))
            plt.axis('off')
            plt.imshow(imgs[j][i+5].reshape((250, 250)), cmap=plt.cm.gray)
    plt.show()


def plot_error(stats_dict):
    plt.figure()
    plt.clf()
    plt.plot(stats_dict['train_error'], 'r-', label="Training Error")
    plt.plot(stats_dict['validation_error'], 'b-', label="Validation Error")
    plt.title('Validation and Training Error')
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()


def show_weights(weights):
    '''weights are expected in the format (filters, channels, x, y)'''
    weights = np.asarray(weights)
    print('Weights Max %f, Mean %f, Min %f' %
                      (weights.max(),
                       weights.mean(),
                       weights.min()))
    plt.figure()
    plt.clf()
    for i in range(weights.shape[0]):
        plt.subplot(int(weights.shape[0] / 8) + 1, 8, i+1)
        plt.axis('off')
        plt.imshow(weights[i].reshape(weights.shape[2:]), cmap=plt.cm.gray, interpolation='none')
    plt.show()
