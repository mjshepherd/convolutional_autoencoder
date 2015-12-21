import numpy as np
import theano
import theano.tensor as T
from convpoollayer import ConvPoolLayer
from hiddenlayer import HiddenLayer
from deconvlayer import DeconvLayer

rng = np.random.RandomState(23455)


def build_8D_model(input, batch_size):
    input = input/255
    layer1 = ConvPoolLayer(
        rng,
        input=input,
        name='C1',
        filter_shape=(8, 1, 15, 15),
        input_shape=(batch_size, 1, 250, 250),
        poolsize=(2, 2)
    )
    layer2 = ConvPoolLayer(
        rng,
        input=layer1.output,
        name='C2',
        filter_shape=(24, 8, 9, 9),
        input_shape=layer1.output_shape,
        poolsize=(2, 2)
    )
    layer3 = ConvPoolLayer(
        rng,
        input=layer2.output,
        name='C3',
        filter_shape=(128, 24, 6, 6),
        input_shape=layer2.output_shape,
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    layer4_input = layer3.output.flatten(2)
    layer4 = HiddenLayer(
        rng,
        input=layer4_input,
        n_in=128 * 25 * 25,
        n_out=1024,
        activation=T.tanh,
        name='FC1'
    )
    layer5 = HiddenLayer(
        rng,
        input=layer4.output,
        n_in=1024,
        n_out=128 * 25 * 25,
        activation=T.tanh,
        name='FC2'
    )
    deconv_input = layer5.output.reshape((batch_size, 128, 25, 25))
    layer6 = DeconvLayer(
        conv_layer=layer3,
        rng=rng,
        input=deconv_input,
        shareW=False,
        name='DC3'
    )
    layer7 = DeconvLayer(
        conv_layer=layer2,
        rng=rng,
        input=layer6.output,
        shareW=False,
        name='DC2'
    )
    layer8 = DeconvLayer(
        conv_layer=layer1,
        rng=rng,
        input=layer7.output,
        shareW=False,
        name='DC1',
        activation = T.nnet.sigmoid
    )
    params = layer1.params + layer2.params + layer3.params + layer4.params +\
        layer5.params + layer6.params + layer7.params +\
        layer8.params

    output = layer8.output.flatten(2) * 255

    return output, params, layer1.W


def build_4D_model(input, batch_size):

    layer1 = ConvPoolLayer(
        rng,
        input=input,
        name='C1',
        filter_shape=(16, 1, 51, 51),
        input_shape=(batch_size, 1, 250, 250),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    layer2_input = layer1.output.flatten(2)
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=16 * 100 * 100,
        n_out=500,
        activation=T.tanh,
        name='FC1'
    )
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=500,
        n_out=16 * 100 * 100,
        activation=T.tanh,
        name='FC2'
    )
    deconv_input = layer3.output.reshape((batch_size, 16, 100, 100))
    layer4 = DeconvLayer(
        conv_layer=layer1,
        rng=rng,
        input=deconv_input,
        shareW=False,
        name='DC1'
    )
    params = layer1.params + layer2.params + layer3.params + layer4.params
    output = layer4.output.flatten(2)

    return output, params, layer1.W


def build_6D_connected_model(input, batch_size):

    layer1_input = input.flatten(2) / 255
    layer1 = HiddenLayer(
        rng,
        input=layer1_input,
        n_in=62500,
        n_out=5000,
        activation=T.tanh,
        name='FC1',
        irange=0.001
    )
    layer2 = HiddenLayer(
        rng,
        input=layer1.output,
        n_in=5000,
        n_out=5000,
        activation=T.tanh,
        name='FC2',
        irange=0.001
    )
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=5000,
        n_out=1000,
        activation=T.tanh,
        name='FC3',
        irange=0.001
    )
    layer4 = HiddenLayer(
        rng,
        input=layer3.output,
        n_in=1000,
        n_out=5000,
        activation=T.tanh,
        name='FC4',
        irange=0.001
    )
    layer5 = HiddenLayer(
        rng,
        input=layer4.output,
        n_in=5000,
        n_out=5000,
        activation=T.tanh,
        name='FC5',
        irange=0.001
    )
    layer6 = HiddenLayer(
        rng,
        input=layer5.output,
        n_in=5000,
        n_out=62500,
        activation=T.nnet.sigmoid,
        name='FC6',
        irange=0.001
    )

    params = layer1.params + layer2.params + layer3.params + layer4.params +\
        layer5.params + layer6.params
    output = layer6.output.flatten(2) * 255

    return output, params, layer1.W


def build_10D_model(input, batch_size):
    # Convolution layers
    layer1 = ConvPoolLayer(
        rng,
        input=input,
        filter_shape=(8, 1, 15, 15),
        input_shape=(batch_size, 1, 250, 250),
        poolsize=(2, 2),
        name='C1'
    )

    layer2 = ConvPoolLayer(
        rng,
        input=layer1.output,
        filter_shape=(24, 8, 9, 9),
        input_shape=layer1.output_shape,
        poolsize=(2, 2),
        name='C2'
    )

    layer3 = ConvPoolLayer(
        rng,
        input=layer2.output,
        filter_shape=(64, 24, 6, 6),
        input_shape=layer2.output_shape,
        poolsize=(2, 2),
        name='C3'
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    layer4_input = layer3.output.flatten(2)
    layer4 = HiddenLayer(
        rng,
        input=layer4_input,
        n_in=64 * 25 * 25,
        n_out=1024,
        activation=T.tanh,
        name='FC1'
    )

    layer5 = HiddenLayer(
        rng,
        input=layer4.output,
        n_in=1024,
        n_out=512,
        activation=T.tanh,
        name='FC2'
    )
    layer6 = HiddenLayer(
        rng,
        input=layer5.output,
        n_in=512,
        n_out=1024,
        activation=T.tanh,
        name='FC3'
    )

    layer7 = HiddenLayer(
        rng,
        input=layer6.output,
        n_in=1024,
        n_out=64 * 25 * 25,
        activation=T.tanh,
        name='FC4'
    )

    ###################
    #  Deconvolution  #
    ###################
    deconv_input = layer7.output.reshape((batch_size, 64, 25, 25))
    layer8 = DeconvLayer(
        conv_layer=layer3,
        rng=rng,
        input=deconv_input,
        shareW=True,
        name='DC3'
    )

    layer9 = DeconvLayer(
        conv_layer=layer2,
        rng=rng,
        input=layer8.output,
        shareW=True,
        name='DC2'
    )

    layer10 = DeconvLayer(
        conv_layer=layer1,
        rng=rng,
        input=layer9.output,
        shareW=True,
        name='DC1'
    )

    params = layer1.params + layer2.params + layer3.params + layer4.params + \
        layer5.params + layer6.params + \
        layer7.params + layer8.params + layer9.params + layer10.params
    output = layer10.output.flatten(2)

    return output, params, layer1.W + layer2.W + layer3.W
