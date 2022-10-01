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

# 84.03
# class SmallConvolutionalNetwork(Module):
#     def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
#         self.net = sequential(
#             ########## TODO: ##########
#             ConvLayer2D(3, 5, 8, 1, 1, name = "conv1"),
#             leaky_relu(name="lr1"),
#             ConvLayer2D(8, 6, 4, 1, 1, name = "conv2"),
#             leaky_relu(name="lr2"),
#             MaxPoolingLayer(3, 1, name = "pool2"),
#             flatten(name = "flatten1"),
#             fc(2500, 10, 0.02, name="fc1"),
#             leaky_relu(name="lr3"),
#             fc(10, 10, 0.02, name="fc2")
#             ########### END ###########
#         )

# 82.xx
# class SmallConvolutionalNetwork(Module):
#     def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
#         self.net = sequential(
#             ########## TODO: ##########
#             ConvLayer2D(3, 3, 16, 1, 0, name = "conv1"),
#             leaky_relu(name="lr1"),
#             MaxPoolingLayer(3, 2, name = "pool1"),
#             ConvLayer2D(16, 3, 8, 1, 0, name = "conv2"),
#             leaky_relu(name="lr2"),
#             MaxPoolingLayer(3, 1, name = "pool2"),
#             flatten(name = "flatten1"),
#             fc(800, 10, 0.02, name="fc1"),
#             leaky_relu(name="lr3"),
#             fc(10, 10, 0.02, name="fc2")
#             ########### END ###########
#         )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(3, 3, 32, 1, 0, name = "conv1"),
            leaky_relu(name="lr1"),
            MaxPoolingLayer(3, 2, name = "pool1"),
            ConvLayer2D(32, 3, 32, 1, 0, name = "conv2"),
            leaky_relu(name="lr2"),
            MaxPoolingLayer(3, 1, name = "pool2"),
            flatten(name = "flatten1"),
            fc(3200, 10, 0.02, name="fc1"),
            leaky_relu(name="lr3"),
            fc(10, 10, 0.02, name="fc2")
            ########### END ###########
        )
