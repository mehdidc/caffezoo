# VGG-19, 19-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/3785162f95cd2d5fee77
# License: non-commercial use only

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer


#from lasagne.layers.conv import Conv2DLayer as ConvLayer
#from lasagne.layers import Pool2DLayer as PoolLayer

#from lasagne.layers.corrmm import Conv2DMMLayer as ConvLayer
#from lasagne.layers import Pool2DLayer as PoolLayer

from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayer

from lasagne.nonlinearities import softmax

from collections import OrderedDict

import cPickle as pickle

from lasagne import layers

import theano.tensor as T
import theano
import numpy as np

from lasagne.nonlinearities import linear, rectify
from lasagne.layers import NonlinearityLayer

from skimage.transform import resize

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

    net['conv3_2'] = ConvLayer(net['conv3_1/relu'], 256, 3, pad=1, name="conv3_2")
    net['conv3_2/relu'] = NonlinearityLayer(net['conv3_2'], nonlinearity=rectify)

    net['conv3_3'] = ConvLayer(net['conv3_2/relu'], 256, 3, pad=1, name="conv3_3")
    net['conv3_3/relu'] = NonlinearityLayer(net['conv3_3'], nonlinearity=rectify)


    net['conv3_4'] = ConvLayer(net['conv3_3/relu'], 256, 3, pad=1, name="conv3_4")
    net['conv3_4/relu'] = NonlinearityLayer(net['conv3_4'], nonlinearity=rectify)


    net['pool3'] = PoolLayer(net['conv3_4/relu'], 2, name="pool3")

    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, name="conv4_1")
    net['conv4_1/relu'] = NonlinearityLayer(net['conv4_1'], nonlinearity=rectify)

    net['conv4_2'] = ConvLayer(net['conv4_1/relu'], 512, 3, pad=1, name="conv4_2")
    net['conv4_2/relu'] = NonlinearityLayer(net['conv4_2'], nonlinearity=rectify)

    net['conv4_3'] = ConvLayer(net['conv4_2/relu'], 512, 3, pad=1, name="conv4_3")
    net['conv4_3/relu'] = NonlinearityLayer(net['conv4_3'], nonlinearity=rectify)

    net['conv4_4'] = ConvLayer(net['conv4_3/relu'], 512, 3, pad=1, name="conv4_4")
    net['conv4_4/relu'] = NonlinearityLayer(net['conv4_4'], nonlinearity=rectify)

    net['pool4'] = PoolLayer(net['conv4_4/relu'], 2, name="pool4")

    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, name="conv5_1")
    net['conv5_1/relu'] = NonlinearityLayer(net['conv5_1'], nonlinearity=rectify)

    net['conv5_2'] = ConvLayer(net['conv5_1/relu'], 512, 3, pad=1, name="conv5_2")
    net['conv5_2/relu'] = NonlinearityLayer(net['conv5_2'], nonlinearity=rectify)

    net['conv5_3'] = ConvLayer(net['conv5_2/relu'], 512, 3, pad=1, name="conv5_3")
    net['conv5_3/relu'] = NonlinearityLayer(net['conv5_3'], nonlinearity=rectify)

    net['conv5_4'] = ConvLayer(net['conv5_3/relu'], 512, 3, pad=1, name="conv5_4")
    net['conv5_4/relu'] = NonlinearityLayer(net['conv5_4'], nonlinearity=rectify)

    net['pool5'] = PoolLayer(net['conv5_4/relu'], 2, name="pool5")

    net['fc6'] = DenseLayer(net['pool5'], num_units=4096, name="fc6")
    net['fc7'] = DenseLayer(net['fc6'], num_units=4096, name="fc7")
    net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None, name="fc8")
    net['prob'] = NonlinearityLayer(net['fc8'], softmax, name="prob")
    return net

def concat(layers):
    layers = [layer.reshape((layer.shape[0], np.prod(layer.shape[1:]))) for layer in layers]
    return np.concatenate(layers, axis=1)

def preprocess(mv, img):
    return (img-mv).transpose((0, 3, 1, 2)).astype(np.float32)


class VGG(object):
    MODEL_FILENAME = "/home/mcherti/work/data/zoo/vgg19.pkl"

    def __init__(self, layer_names=None,
                 aggregate_function=concat,
                 model_filename=MODEL_FILENAME,
                 batch_size=100, resize=(224, 224)):
        if layer_names is None:
            layer_names = ["pool3"]
        self.layer_names = layer_names
        self.model_filename = model_filename
        self.mean_value = None
        self.aggregate_function = aggregate_function
        self.batch_size = batch_size
        self.resize = resize

        self._loaded = False
        self._predict_layers = None

    def transform(self, X):
        nb_batches = X.shape[0] / self.batch_size
        if (X.shape[0] % self.batch_size):
            nb_batches += 1
        last = 0
        O = []
        for i in range(nb_batches):
            first = last
            last += self.batch_size

            X_batch = X[first:last]
            if self.resize is not False:
                X_batch_rescaled = np.empty((X_batch.shape[0], self.resize[0], self.resize[1], 3))
                for j in range(X_batch.shape[1]):
                    X_batch_rescaled[j] = resize(X_batch[j], (self.resize[0], self.resize[1]), preserve_range=True)
            else:
                X_batch_rescaled = X_batch
            O.append(self.aggregate_function(self._predict_layers(preprocess(X_batch_rescaled, self.mean_value))))
        return np.concatenate(O, axis=0)


    def fit(self, X, y=None):
        self._load()
        return self

    def _load(self):
        net = build_model()
        model_data = pickle.load(open(self.model_filename))
        values = model_data['param values']
        layers.set_all_param_values(net['prob'], values)

        if "mean value" in model_data:
            mean_value = (model_data["mean value"])
        else:
            mean_value = np.array([104.0, 116.0, 122.0])
        self.mean_value = mean_value

        X = T.tensor4()
        layer_values = [layers.get_output(net[layer], X) for layer in self.layer_names]
        self._predict_layers =  theano.function([X], layer_values)

        self._loaded = True
        self.all_layer_names = net.keys()
        self._net = net
