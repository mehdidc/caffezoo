import cPickle as pickle
from lasagne import layers
import numpy as np
import theano.tensor as T

# BLVC Googlenet, model from the paper:
# "Going Deeper with Convolutions"
# Original source:
# https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
# License: unrestricted use

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/blvc_googlenet.pkl
import os
#os.environ["THEANO_FLAGS"] = "device=gpu"
import theano

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import GlobalPoolLayer


from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayer

#from lasagne.layers.corrmm import Conv2DMMLayer as ConvLayer
#from lasagne.layers.pool import MaxPool2DLayer as PoolLayer

from lasagne.layers import Pool2DLayer as PoolLayerDefault


from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.nonlinearities import softmax, linear, rectify

from collections import OrderedDict

from skimage.transform import resize



#from theano.sandbox.cuda.blas import GpuCorrMM

def build_inception_module(name, input_layer, nfilters, pool_mode='max'):
    # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
    net = OrderedDict()
    net['pool'] = PoolLayer(input_layer, pool_size=3, stride=1, pad=1)
    net['pool_proj'] = ConvLayer(net['pool'], nfilters[0], 1)

    net['1x1'] = ConvLayer(input_layer, nfilters[1], 1, nonlinearity=linear)
    net['1x1/relu'] = NonlinearityLayer(net['1x1'], nonlinearity=rectify)

    net['3x3_reduce'] = ConvLayer(input_layer, nfilters[2], 1, nonlinearity=linear)
    net['3x3_reduce/relu'] = NonlinearityLayer(net['3x3_reduce'], nonlinearity=rectify)

    net['3x3'] = ConvLayer(net['3x3_reduce/relu'], nfilters[3], 3, pad=1, nonlinearity=linear)
    net['3x3/relu'] = NonlinearityLayer(net['3x3'], nonlinearity=rectify)

    net['5x5_reduce'] = ConvLayer(input_layer, nfilters[4], 1, nonlinearity=linear)
    net['5x5_reduce/relu'] = NonlinearityLayer(net['5x5_reduce'], nonlinearity=rectify)

    net['5x5'] = ConvLayer(net['5x5_reduce/relu'], nfilters[5], 5, pad=2, nonlinearity=linear)
    net['5x5/relu'] = NonlinearityLayer(net['5x5'], nonlinearity=rectify)

    net['output'] = ConcatLayer([
        net['1x1/relu'],
        net['3x3/relu'],
        net['5x5/relu'],
        net['pool_proj'],
        ])
    return {'{}/{}'.format(name, k): v for k, v in net.items()}


def build_model(pool_mode='max'):
    net = OrderedDict()
    net['input'] = InputLayer((None, 3, None, None), name="input")
    net['conv1/7x7_s2'] = ConvLayer(net['input'], 64, 7, stride=2, pad=3,
                                    nonlinearity=linear)

    net['conv1/7x7_s2/relu'] = NonlinearityLayer(net['conv1/7x7_s2'], nonlinearity=rectify)

    net['pool1/3x3_s2'] = PoolLayerDefault(net['conv1/7x7_s2/relu'],
                                           pool_size=3,
                                           stride=2,
                                           ignore_border=False)
    net['pool1/norm1'] = LRNLayer(net['pool1/3x3_s2'], alpha=0.00002, k=1)

    net['conv2/3x3_reduce'] = ConvLayer(net['pool1/norm1'], 64, 1, nonlinearity=linear)
    net['conv2/3x3_reduce/relu'] = NonlinearityLayer(net['conv2/3x3_reduce'], nonlinearity=rectify)

    net['conv2/3x3'] = ConvLayer(net['conv2/3x3_reduce/relu'], 192, 3, pad=1, nonlinearity=linear)
    net['conv2/3x3/relu'] = NonlinearityLayer(net['conv2/3x3'], nonlinearity=rectify)

    net['conv2/norm2'] = LRNLayer(net['conv2/3x3/relu'], alpha=0.00002, k=1)
    net['pool2/3x3_s2'] = PoolLayer(net['conv2/norm2'], pool_size=3, stride=2)

    net.update(build_inception_module('inception_3a',
                                      net['pool2/3x3_s2'],
                                      [32, 64, 96, 128, 16, 32]))
    net.update(build_inception_module('inception_3b',
                                      net['inception_3a/output'],
                                      [64, 128, 128, 192, 32, 96]))
    net['pool3/3x3_s2'] = PoolLayer(net['inception_3b/output'],
                                    pool_size=3, stride=2)

    net.update(build_inception_module('inception_4a',
                                      net['pool3/3x3_s2'],
                                      [64, 192, 96, 208, 16, 48]))
    net.update(build_inception_module('inception_4b',
                                      net['inception_4a/output'],
                                      [64, 160, 112, 224, 24, 64]))
    net.update(build_inception_module('inception_4c',
                                      net['inception_4b/output'],
                                      [64, 128, 128, 256, 24, 64]))
    net.update(build_inception_module('inception_4d',
                                      net['inception_4c/output'],
                                      [64, 112, 144, 288, 32, 64]))
    net.update(build_inception_module('inception_4e',
                                      net['inception_4d/output'],
                                      [128, 256, 160, 320, 32, 128]))
    net['pool4/3x3_s2'] = PoolLayer(net['inception_4e/output'],
                                    pool_size=3, stride=2)

    net.update(build_inception_module('inception_5a',
                                      net['pool4/3x3_s2'],
                                      [128, 256, 160, 320, 32, 128]))
    net.update(build_inception_module('inception_5b',
                                      net['inception_5a/output'],
                                      [128, 384, 192, 384, 48, 128]))

    net['pool5/7x7_s1'] = GlobalPoolLayer(net['inception_5b/output'])
    net['loss3/classifier'] = DenseLayer(net['pool5/7x7_s1'],
                                         num_units=1000,
                                         nonlinearity=linear)
    net['prob'] = NonlinearityLayer(net['loss3/classifier'],
                                    nonlinearity=softmax)
    for k, v in net.items():
        net[k].name = k
    return net


def concat(layers):
    layers = [layer.reshape((layer.shape[0], np.prod(layer.shape[1:]))) for layer in layers]
    return np.concatenate(layers, axis=1)

def preprocess(mv, img):
    return (img-mv).transpose((0, 3, 1, 2)).astype(np.float32)

class GoogleNet(object):
    MODEL_FILENAME = "/home/mcherti/work/data/zoo/blvc_googlenet.pkl"

    def __init__(self, layer_names=None,
                 aggregate_function=concat,
                 model_filename=MODEL_FILENAME,
                 batch_size=100, resize=(224, 224)):
        if layer_names is None:
            layer_names = ["inception_3b/output"]
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
