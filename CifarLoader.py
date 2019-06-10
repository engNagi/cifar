import pickle as cPickle
import numpy as np
import os


class CifarLoader(object):
    def __init__(self,source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None

    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack(d["data"] for d in data)
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float)/255
        self.labels = one_hot(np.hstack([d["labels"] for d in data]), 10)
        return self

    def next_batch(self, batch_size):
        x, y = self.images[self._i: self._i+batch_size], self.labels[self._i: self._i+batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y


DATA_PATH = "/home/nagi/Desktop/thesis/1stWeek/CIFAR10/cifar-10-batches-py"

# The unpickle() function returns a dict with fields data and labels
def unpickle(file):
    with open(os.path.join(DATA_PATH, file), 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

#one_hot() recodes the labels from integers (in
#the range 0 to 9) to vectors of length 10, containing all 0s except for a 1 at the position
#of the label.
def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n,vals))
    out[range(n), vec] = 1
    return out

