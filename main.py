import os
import sys
import cPickle
import timeit

import utils
import theano
import theano.tensor as T
import numpy as np
import models

from loaddata import read_dataset

batch_size = 1
learning_rate = 0.001
n_epochs = 500



import pdb


def build_model(dataset):

    # instantiate 4D tensor for input
    input = T.tensor4(name='input')
    target = T.tensor4(name='target')

    output, params, layer1Weights = models.build_8D_model(input, batch_size)

    flat_target = target.flatten(2)

    mse = ((flat_target - output) ** 2).mean()

    gradients = T.grad(mse, params)
    error = mse

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, gradients)
    ]
    print("Compiling function")
    predict_func = theano.function(inputs=[input], outputs=[output])
    train_func = theano.function(
        [input, target],
        outputs=[output, error],
        updates=updates,
    )
    valid_func = theano.function(inputs=[input, target],
                                 outputs=[error])
    get_weights = theano.function(inputs=[], outputs=[layer1Weights])
    print("Done!")

    return predict_func, train_func, valid_func, get_weights


def train_model(train_func, valid_func, train_set, valid_set):
    n_train_batches = train_set.shape[0]
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 1000  # look as this many examples regardless
    patience_increase = 100  # wait this much longer when a new best is found
    improvement_threshold = 0.998
    validation_frequency = min(n_train_batches, patience / 2)

    # Keep track of statistics
    validation_error = []
    train_error = []

    # Initialize some variables
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % train_set.shape[0] == 0:
                print 'training @ iter = ', iter
            training_batch = np.repeat(train_set[minibatch_index:minibatch_index + 1],
                                       batch_size, axis=0)

            reconstruction, train_cost = train_func(
                utils.create_random_hole(training_batch), training_batch)
            if (iter + 1) % validation_frequency == 0:
                this_validation_loss = utils.get_valid_loss(
                    valid_set, valid_func, batch_size)

                # print some reconstruction stats
                print('Reconstruction Max %f, Mean %f, Min %f' %
                      (reconstruction.max(),
                       reconstruction.mean(),
                       reconstruction.min()))
                # Record some stats
                validation_error.append(this_validation_loss)
                train_error.append(train_cost)

                print('epoch %i, minibatch %i/%i, validation error %f ' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss))

                # if we got the best validation score until now
                if this_validation_loss <= best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f' %
          (best_validation_loss, best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    return dict(train_error=train_error, validation_error=validation_error)

if __name__ == '__main__':
    train_set = read_dataset(which='tiny')
    valid_set = read_dataset(which='tiny')

    predict_func, train_func, valid_func, get_weights = build_model(train_set)

    stats = train_model(train_func=train_func, valid_func=valid_func,
                        train_set=train_set, valid_set=valid_set)

    model = dict(stats=stats, predict=predict_func, weights=get_weights())
    # pickle predict_func
    f = file('models/tanh_attempt.pkl', 'wb')
    cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
