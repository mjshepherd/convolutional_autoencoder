import numpy
import theano
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv


class ConvPoolLayer(object):

    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, input_shape, poolsize=(2, 2), border_mode='valid',
                 name='Default_Convolution'):
        """

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape input_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type input_shape: tuple or list of length 4
        :param input_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        print('Building layer: ' + name)

        assert input_shape[1] == filter_shape[1]
        self.input = input
        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.poolsize = poolsize
        self.output_shape = (input_shape[0],
                             filter_shape[0],
                             (input_shape[2] -
                              filter_shape[2] + 1) / poolsize[0],
                             (input_shape[3] - filter_shape[3] + 1) / poolsize[1])
        print("ConvLayer Output: " + str(self.output_shape))

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=input_shape,
            border_mode=border_mode
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = theano.tensor.tanh(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def relu(self, x):
        return theano.tensor.switch(x < 0, 0.01*x, x)
