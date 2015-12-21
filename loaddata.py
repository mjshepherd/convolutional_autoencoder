import os
import numpy as np


def read_dataset(which='small_test_10', corrupt=False):
    path = os.path.expandvars("data/")
    file = which + '.npz'
    fullPath = path + file
    set = np.load(fullPath)
    y = np.asarray(
        np.hsplit(set['arr_0'], set['arr_1'])).astype('float32')
    y = np.expand_dims(y, axis=1)

    if corrupt:
        print('Corrupting training set')
        corrupted = create_random_hole(y)
        return y, corrupted

    return y
