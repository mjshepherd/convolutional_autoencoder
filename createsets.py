import numpy as np
from scipy import ndimage
import sys
import os


import pdb


def create_set(path, out, size):
    result = np.empty((250, 0))
    fileNames = np.random.choice(
        [x for x in os.listdir(path)], size=size, replace=False)
    for fileName in fileNames:
        img = ndimage.imread(path + fileName)
        result = np.hstack((result, img))
    np.savez(out, result, size)


def create_sets(path, basename, train_size, valid_size=None, test_size=10):
    '''creates three sets training, test, valid'''

    if valid_size is None:
        valid_size = int(0.2 * train_size)
    train_set = np.empty((250, 0))
    valid_set = np.empty((250, 0))
    test_set = np.empty((250, 0))

    total_images = train_size + valid_size + test_size

    fileNames = np.random.choice(
        [x for x in os.listdir(path)], size=total_images, replace=False)
    for idx, fileName in enumerate(fileNames):
        img = ndimage.imread(path + fileName)
        if idx < train_size:
            train_set = np.hstack((train_set, img))
        elif idx < valid_size + train_size:
            valid_set = np.hstack((valid_set, img))
        else:
            test_set = np.hstack((test_set, img))

    train_name = out + '_train_' + str(train_size)
    valid_name = out + '_valid_' + str(valid_size)
    test_name = out + '_test_' + str(test_size)

    np.savez(train_name, train_set, train_size)
    np.savez(valid_name, valid_set, valid_size)
    np.savez(test_name, test_set, test_size)


def load_set(name):
    set = np.load(name)
    return np.asarray(np.hsplit(set['arr_0'], set['arr_1']))

if __name__ == '__main__':
    path = sys.argv[1]
    out = sys.argv[2]
    size = int(sys.argv[3])
    create_sets(path, out, size)
