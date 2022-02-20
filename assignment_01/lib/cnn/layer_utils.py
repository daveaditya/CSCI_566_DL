from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class sequential(object):
    def __init__(self, *args):
        """
        Sequential Object to serialize the NN layers
        Please read this code block and understand how it works
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        # process the parameters layer by layer
        for layer_cnt, layer in enumerate(args):
            for n, v in layer.params.items():
                self.params[n] = v
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)

    def assign(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        # return the parameters by name
        return self.params[name]

    def get_grads(self, name):
        # return the gradients by name
        return self.grads[name]

    def gather_params(self):
        """
        Collect the parameters of every submodules
        """
        for layer in self.layers:
            for n, v in layer.params.items():
                self.params[n] = v

    def gather_grads(self):
        """
        Collect the gradients of every submodules
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] = v

    def load(self, pretrained):
        """
        Load a pretrained model by names
        """
        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.items():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))


class ConvLayer2D(object):
    def __init__(self, input_channels, kernel_size, number_filters, stride=1, padding=0, init_scale=0.02, name="conv"):

        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.number_filters = number_filters
        self.stride = stride
        self.padding = padding

        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(
            kernel_size, kernel_size, input_channels, number_filters
        )
        self.params[self.b_name] = np.zeros(number_filters)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None

    def get_output_size(self, input_size):
        """
        :param input_size - 4-D shape of input image tensor (batch_size, in_height, in_width, in_channels)
        :output a 4-D shape of the output after passing through this layer (batch_size, out_height, out_width, out_channels)
        """
        output_shape = [None, None, None, None]
        #############################################################################
        # TODO: Implement the calculation to find the output size given the         #
        # parameters of this convolutional layer.                                   #
        #############################################################################
        batch_size, input_height, input_width, _ = input_size

        output_shape[0] = batch_size
        output_shape[1] = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_shape[2] = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_shape[3] = self.number_filters
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output_shape

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        output_shape = self.get_output_size(img.shape)
        _, input_height, input_width, _ = img.shape
        _, output_height, output_width, _ = output_shape

        #############################################################################
        # TODO: Implement the forward pass of a single fully connected layer.       #
        # Store the results in the variable "output" provided above.                #
        #############################################################################
        # store for the batch of images for backprop
        self.meta = img

        # pad the input according to self.padding (see np.pad)
        img_padded = (
            img
            if self.padding == 0
            else np.pad(
                img,
                ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                "constant",
                constant_values=(0, 0),
            )
        )

        # initialize output
        output = np.zeros(output_shape)

        weights = self.params[self.w_name]
        bias = self.params[self.b_name]

        # Start filling from the height of the output
        for h in range(output_height):

            # Calculate the coordinates of the current window top and bottom
            top, bottom = h * self.stride, h * self.stride + self.kernel_size

            # Followed by the width
            # i.e. top-bottom, left-right manner
            for w in range(output_width):

                # Calculate the coordinates of the current window left and right
                left, right = w * self.stride, w * self.stride + self.kernel_size

                # Get the image slice coresponding to the current window coordinates
                img_slices = img_padded[:, top:bottom, left:right, :]
                img_slices_reshaped = np.expand_dims(img_slices, -1)

                output[:, h, w, :] = np.sum( np.multiply(weights, img_slices_reshaped) , axis = (1,2,3)) + bias
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return output

    def backward(self, dprev):
        img = self.meta
        if img is None:
            raise ValueError("No forward function called before for this module!")

        dimg, self.grads[self.w_name], self.grads[self.b_name] = None, None, None

        #############################################################################
        # TODO: Implement the backward pass of a single convolutional layer.        #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        # Get the dimensions out the input and the derivative
        batch_size, input_height, input_width, input_no_channels = img.shape
        _, output_height, output_width, output_no_channels = dprev.shape

        # Initialize dimg, dW, and db
        dimg = np.zeros((batch_size, input_height, input_width, input_no_channels))
        dW = np.zeros((self.kernel_size, self.kernel_size, input_no_channels, output_no_channels))
        db = np.zeros(output_no_channels)

        # Pad the img and dprev
        img_padded = (
            img
            if self.padding == 0
            else np.pad(
                img,
                ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                "constant",
                constant_values=(0, 0),
            )
        )
        dimg_padded = (
            dimg
            if self.padding == 0
            else np.pad(
                dimg,
                ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                "constant",
                constant_values=(0, 0),
            )
        )


        # Start from the height of the output
        for h in range(output_height):

            # Calculate the coordinates of the current window top and bottom
            top, bottom = h * self.stride, h * self.stride + self.kernel_size

            # Towards the width of the output
            # i.e. Move from top-to-bottom and left-to-right
            for w in range(output_width):

                # Calculate the coordinates of the current window left and right
                left, right = w * self.stride, w * self.stride + self.kernel_size

                # Get the image slice of pointed by the window coordinates
                img_padded_slices = img_padded[: , top:bottom, left:right, :]
                img_padded_slices_reshaped = img_padded_slices.reshape(batch_size, self.kernel_size, self.kernel_size, img.shape[-1], 1)
                
                dprev_reshaped = dprev[:, h, w, :].reshape(batch_size, 1, 1, 1, self.number_filters)

                # Calculate the derivative for the current image slice window
                dimg_padded[:, top:bottom, left:right, :] += np.sum(self.params[self.w_name] * dprev_reshaped, axis = 4)

                # Calculate derivative of the weight
                dW += np.sum(img_padded_slices_reshaped * dprev_reshaped, axis = 0)

                # Calculate derivative of the bias
                db += np.sum(dprev_reshaped, axis = (0,1,2,3))

        # Set the dimg for current image without padding
        if self.padding > 0:
            dimg = dimg_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            dimg = dimg_padded

        self.grads[self.w_name] = dW
        self.grads[self.b_name] = db
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        self.meta = None
        return dimg


class MaxPoolingLayer(object):
    def __init__(self, pool_size, stride, name):
        self.name = name
        self.pool_size = pool_size
        self.stride = stride
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        #############################################################################
        # TODO: Implement the forward pass of a single maxpooling layer.            #
        # Store your results in the variable "output" provided above.               #
        #############################################################################
        # Store for backward pass
        self.meta = img

        # Calculate output shape
        batch_size, input_height, input_width, no_channels = img.shape

        # Calculate and store the output shape
        output_height = int(1 + (input_height - self.pool_size) / self.stride)
        output_width = int(1 + (input_width - self.pool_size) / self.stride)
        out_channels = no_channels
        output_shape = (batch_size, output_height, output_width, out_channels)

        # Initialize the output
        output = np.zeros(output_shape)

        # Start from the height
        for h in range(output_height):

            # Calulate the current window coordinates top and bottom
            top, bottom = h * self.stride, h * self.stride + self.pool_size

            # Towards the width
            # i.e. from top-to-bottom, left-to-right
            for w in range(output_width):

                # Calulate the current window coordinates left and right
                left, right = w * self.stride, w * self.stride + self.pool_size

                # Get the part of the image acted upon by the current window
                img_slices = img[:, top:bottom, left:right, :]

                output[:, h, w, :] = np.max(img_slices, axis=(1, 2))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output

    def backward(self, dprev):
        img = self.meta

        dimg = np.zeros_like(img)
        _, h_out, w_out, _ = dprev.shape
        h_pool, w_pool = self.pool_size, self.pool_size

        #############################################################################
        # TODO: Implement the backward pass of a single maxpool layer.              #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        # Get the batch size, and respective output dimensions
        batch_size, h_out, w_out, channel_out = dprev.shape

        # Initialize dimg
        dimg = np.zeros_like(img)

        # Start from the height
        for h in range(h_out):

            # Calculate the current window coordinates top and bottom
            top, bottom = h * self.stride, h * self.stride + h_pool

            # Go towards the width
            # i.e. from top-to-bottom and left-to-right
            for w in range(w_out):

                # Calculate the current window coordinates left anf right
                left, right = w * self.stride, w * self.stride + w_pool

                # Get the current image part that is to be worked on
                img_slices = img[:, top:bottom, left:right, :]

                # Get the mask corresponding to the images' slice
                mask = img_slices == np.max(img_slices, axis = (1, 2)).reshape((batch_size, 1, 1, channel_out))

                dimg[:, top:bottom, left:right, :] += mask * dprev[:, h, w, :].reshape(batch_size, 1, 1, channel_out)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dimg
