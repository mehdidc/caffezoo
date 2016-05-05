from skimage.transform import resize
import numpy as np
import pickle
from lasagne import layers
import theano.tensor as T
import theano


def concat(layers):
    layers = [layer.reshape((layer.shape[0], np.prod(layer.shape[1:]))) for layer in layers]
    return np.concatenate(layers, axis=1)


def preprocess(mv, img):
    img = img[:, :, :, ::-1]
    return (img-mv).transpose((0, 3, 1, 2)).astype(np.float32)


def deprocess(mv, img):
    img = img[:, ::-1, :, :]
    (img + mv).transpose((0, 2, 3, 1)).astype(np.float32)


class BaseModel(object):
    default_layers = []
    default_filename = ""

    def __init__(self, layer_names=None,
                 aggregate_function=concat,
                 model_filename=None,
                 batch_size=100, resize=(224, 224),
                 input_size=None):
        if layer_names is None:
            layer_names = self.default_layers
        self.layer_names = layer_names
        if model_filename is None:
            model_filename = self.default_filename
        self.model_filename = model_filename
        self.mean_value = None
        self.aggregate_function = aggregate_function
        self.batch_size = batch_size
        self.resize = resize
        self.input_size = input_size
        self._loaded = False
        self._predict_layers = None

    def transform(self, X):
        nb_batches = len(X) / self.batch_size
        if (len(X) % self.batch_size):
            nb_batches += 1
        last = 0
        O = []
        for i in range(nb_batches):
            first = last
            last += self.batch_size

            X_batch = X[first:last]
            X_batch = self.preprocess(X_batch)
            O.append(self.aggregate_function(self._predict_layers(X_batch)))
        return np.concatenate(O, axis=0)

    def preprocess(self, X):
        if self.resize != False:
            X_rescaled = np.empty((len(X), self.resize[0], self.resize[1], 3))
            for j in range(len(X)):
                X_rescaled[j] = resize(X[j], (self.resize[0], self.resize[1]), preserve_range=True)
        else:
            X_rescaled = X
        return preprocess(self.mean_value, X_rescaled)

    def deprocess(self, X):
        return deprocess(self.mean_value, X)

    def fit(self, X, y=None):
        self._load()
        return self

    def _load(self):
        net = self._build_model(self.input_size)
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
