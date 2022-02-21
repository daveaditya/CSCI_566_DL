from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(3, 3, 3, name="conv"),
            MaxPoolingLayer(2, 2, name="maxp"),
            flatten(name="flatten"),
            fc(27, 5, init_scale=0.02, name="fc")
            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(3, 5, 32, name="conv1"),
            leaky_relu(name="lrelu1"),
            ConvLayer2D(32, 5, 64, name="conv2"),
            leaky_relu(name="lrelu2"),
            MaxPoolingLayer(2, 2, name="maxp1"),
            flatten(name="flatten"),
            fc(9216, 9216, name="fc1"),
            leaky_relu(name="lrelu3"),
            fc(9216, 10, name="fc2"),
            ########### END ###########
        )
