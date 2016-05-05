# VGG-19, 19-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/3785162f95cd2d5fee77
# License: non-commercial use only

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl
import os

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer


# from lasagne.layers.conv import Conv2DLayer as ConvLayer
# from lasagne.layers import Pool2DLayer as PoolLayer

try:
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
    from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayer
except Exception:
    try:
        from lasagne.layers.corrmm import Conv2DMMLayer as ConvLayer
    except Exception:
        from lasagne.layers import Conv2DLayer as ConvLayer
    from lasagne.layers import Pool2DLayer as PoolLayer



from lasagne.nonlinearities import softmax

from collections import OrderedDict

from lasagne.nonlinearities import linear, rectify

from .base import BaseModel


def build_model(pool_mode='max'):
    net = OrderedDict()
    net['input'] = InputLayer((None, 3, 224, 224), name='input')

    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1, name='conv1_1', nonlinearity=linear)
    net['conv1_1/relu'] = NonlinearityLayer(net['conv1_1'], nonlinearity=rectify)

    net['conv1_2'] = ConvLayer(net['conv1_1/relu'], 64, 3, pad=1, name="conv1_2", nonlinearity=linear)
    net['conv1_2/relu'] = NonlinearityLayer(net['conv1_2'], nonlinearity=rectify)

    net['pool1'] = PoolLayer(net['conv1_2/relu'], 2, name="pool1")

    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, name="conv2_1", nonlinearity=linear)
    net['conv2_1/relu'] = NonlinearityLayer(net['conv2_1'], nonlinearity=rectify)

    net['conv2_2'] = ConvLayer(net['conv2_1/relu'], 128, 3, pad=1, name="conv2_2", nonlinearity=linear)
    net['conv2_2/relu'] = NonlinearityLayer(net['conv2_2'], nonlinearity=rectify)

    net['pool2'] = PoolLayer(net['conv2_2/relu'], 2, name="pool2")

    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, name="conv3_1", nonlinearity=linear)
    net['conv3_1/relu'] = NonlinearityLayer(net['conv3_1'], nonlinearity=rectify)

    net['conv3_2'] = ConvLayer(net['conv3_1/relu'], 256, 3, pad=1, name="conv3_2", nonlinearity=linear)
    net['conv3_2/relu'] = NonlinearityLayer(net['conv3_2'], nonlinearity=rectify)

    net['conv3_3'] = ConvLayer(net['conv3_2/relu'], 256, 3, pad=1, name="conv3_3", nonlinearity=linear)
    net['conv3_3/relu'] = NonlinearityLayer(net['conv3_3'], nonlinearity=rectify)


    net['conv3_4'] = ConvLayer(net['conv3_3/relu'], 256, 3, pad=1, name="conv3_4", nonlinearity=linear)
    net['conv3_4/relu'] = NonlinearityLayer(net['conv3_4'], nonlinearity=rectify)


    net['pool3'] = PoolLayer(net['conv3_4/relu'], 2, name="pool3")

    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, name="conv4_1", nonlinearity=linear)
    net['conv4_1/relu'] = NonlinearityLayer(net['conv4_1'], nonlinearity=rectify)

    net['conv4_2'] = ConvLayer(net['conv4_1/relu'], 512, 3, pad=1, name="conv4_2", nonlinearity=linear)
    net['conv4_2/relu'] = NonlinearityLayer(net['conv4_2'], nonlinearity=rectify)

    net['conv4_3'] = ConvLayer(net['conv4_2/relu'], 512, 3, pad=1, name="conv4_3", nonlinearity=linear)
    net['conv4_3/relu'] = NonlinearityLayer(net['conv4_3'], nonlinearity=rectify)

    net['conv4_4'] = ConvLayer(net['conv4_3/relu'], 512, 3, pad=1, name="conv4_4", nonlinearity=linear)
    net['conv4_4/relu'] = NonlinearityLayer(net['conv4_4'], nonlinearity=rectify)

    net['pool4'] = PoolLayer(net['conv4_4/relu'], 2, name="pool4")

    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, name="conv5_1", nonlinearity=linear)
    net['conv5_1/relu'] = NonlinearityLayer(net['conv5_1'], nonlinearity=rectify)

    net['conv5_2'] = ConvLayer(net['conv5_1/relu'], 512, 3, pad=1, name="conv5_2", nonlinearity=linear)
    net['conv5_2/relu'] = NonlinearityLayer(net['conv5_2'], nonlinearity=rectify)

    net['conv5_3'] = ConvLayer(net['conv5_2/relu'], 512, 3, pad=1, name="conv5_3", nonlinearity=linear)
    net['conv5_3/relu'] = NonlinearityLayer(net['conv5_3'], nonlinearity=rectify)

    net['conv5_4'] = ConvLayer(net['conv5_3/relu'], 512, 3, pad=1, name="conv5_4", nonlinearity=linear)
    net['conv5_4/relu'] = NonlinearityLayer(net['conv5_4'], nonlinearity=rectify)

    net['pool5'] = PoolLayer(net['conv5_4/relu'], 2, name="pool5")

    net['fc6'] = DenseLayer(net['pool5'], num_units=4096, name="fc6", nonlinearity=linear)
    net['fc6/relu'] = NonlinearityLayer(net['fc6'], nonlinearity=rectify)
    net['fc7'] = DenseLayer(net['fc6/relu'], num_units=4096, name="fc7", nonlinearity=linear)
    net['fc7/relu'] = NonlinearityLayer(net['fc7'], nonlinearity=rectify)
    net['fc8'] = DenseLayer(net['fc7/relu'], num_units=1000, nonlinearity=linear, name="fc8")
    net['prob'] = NonlinearityLayer(net['fc8'], softmax, name="prob")
    for k, v in net.items():
        net[k].name = k
    return net


class VGG(BaseModel):
    default_filename = os.path.join(os.getenv("DATA_PATH"), "zoo", "vgg19.pkl")
    default_layers = ["pool3"]

    def _build_model(self, input_size):
        return build_model()
