import numpy as np
import pdb


def create_random_hole(imgs, holeSize=25, holeValue=0):
    result = np.copy(imgs)
    num_imgs, num_channels, imX, imY = imgs.shape

    # quad nested loop :|
    for i in range(num_imgs):
        holeX = np.floor(np.random.rand(1) * (imX - holeSize))
        holeY = np.floor(np.random.rand(1) * (imY - holeSize))
        for c in range(num_channels):
            for x in range(holeX, holeX + holeSize):
                for y in range(holeY, holeY + holeSize):
                    result[i][c][x][y] = holeValue
    return result


def add_gaussian_noise(imgs, sigma=10):
    noise = np.random.normal(
        loc=0, scale=sigma, size=imgs.shape).astype('float32')
    return imgs + noise


def get_random_batch(array, size):
    indices = np.random.choice(range(array.shape[0]), size=size, replace=False)
    return np.asarray([array[i] for i in indices])


def get_valid_loss(valid_set, valid_func, batch_size):
    n_batches = valid_set.shape[0] / batch_size
    av_valid_loss = 0
    for i in range(n_batches):
        valid_batch = valid_set[i * batch_size: (i + 1) * batch_size]
        this_validation_loss = valid_func(
            create_random_hole(valid_batch), valid_batch)[0]
        av_valid_loss = ((float(i) / (i + 1)) * av_valid_loss) + \
            ((float(1) / (i + 1)) * this_validation_loss)
    return av_valid_loss
