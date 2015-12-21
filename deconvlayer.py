import numpy
import theano
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv


import pdb


class DeconvLayer(object):

    """A layer to perform unpooling and deconvolution"""

    def __init__(self, rng, input,
                 filter_shape=None, input_shape=None, poolsize=(1, 1),
                 scale=2, border_mode='full', conv_layer=None, shareW=False,
                 name='Default_Deconvolution', activation=None):
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
        self.input = input

        if conv_layer is not None:
            conv_W, conv_B = conv_layer.params
            filter_shape = (conv_layer.filter_shape[1],
                            conv_layer.filter_shape[0],
                            conv_layer.filter_shape[2],
                            conv_layer.filter_shape[3])

            input_shape = (conv_layer.output_shape[0],
                           conv_layer.output_shape[1],
                           conv_layer.output_shape[2] * scale,
                           conv_layer.output_shape[3] * scale)
            print('deconv input shape: ' + str(input_shape))
            print('deconv filter shape: ' + str(filter_shape))
            scale = conv_layer.poolsize[0]
        if shareW:
            self.W = conv_layer.W.dimshuffle(1, 0, 2, 3)[:, :, ::-1, ::-1]
        else:
            assert input_shape[1] == filter_shape[1]
            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = numpy.prod(filter_shape[1:])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #   pooling size
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) *
                       (scale ** 2))
            # initialize weights with random weights
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(
                        low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros(
            (filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        upsampled_out = input.repeat(scale, axis=2).repeat(scale, axis=3)

        # convolve upsampled feature maps with filters
        conv_out = conv.conv2d(
            input=upsampled_out,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=input_shape,
            border_mode=border_mode
        )

        if activation is None:
            self.output = theano.tensor.tanh(
                conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        else:
            self.output = theano.tensor.nnet.sigmoid(
                conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b] if not shareW else [self.b]

        # keep track of model input
        self.input = input

    def relu(self, x):
        return theano.tensor.switch(x < 0, 0.01*x, x)
