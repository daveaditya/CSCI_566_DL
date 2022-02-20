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
        def convolve(slice, W, b):
            """Convolve a single slice

            Args:
                slice (np.ndarray): A slice of image
                W (np.ndarray): The weights / kernel to convolve with
                b (np.ndarray): The bias for the convulation

            Returns:
                np.ndarray: The convolved result
            """
            slice_dot_w = np.multiply(slice, W)
            slice_sum = np.sum(slice_dot_w)
            output = slice_sum + b.astype(float)
            return output

        # store for the batch of images for backprop
        self.meta = img

        # Get the batch size
        batch_size = img.shape[0]

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

        # Convolve over every image
        for i in range(batch_size):

            # Start filling from the height of the output
            for h in range(output_height):

                # Calculate the coordinates of the current window top and bottom
                top, bottom = h * self.stride, h * self.stride + self.kernel_size

                # Followed by the width
                # i.e. top-bottom, left-right manner
                for w in range(output_width):

                    # Calculate the coordinates of the current window left and right
                    left, right = w * self.stride, w * self.stride + self.kernel_size

                    # Loop over the filters
                    for ch in range(self.number_filters):

                        # Get the image slice coresponding to the current window coordinates
                        img_slice = img_padded[i, top:bottom, left:right, :]

                        # Convolve and store in the output
                        output[i, h, w, ch] = convolve(
                            img_slice, self.params[self.w_name][:, :, :, ch], self.params[self.b_name][ch]
                        )
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

        # Iterate over every image in the batch
        for i in range(batch_size):

            # Start from the height of the output
            for h in range(output_height):

                # Calculate the coordinates of the current window top and bottom
                top, bottom = h * self.stride, h * self.stride + self.kernel_size

                # Towards the width of the output
                # i.e. Move from top-to-bottom and left-to-right
                for w in range(output_width):

                    # Calculate the coordinates of the current window left and right
                    left, right = w * self.stride, w * self.stride + self.kernel_size

                    # Calculate for every channel i.e. filter
                    for ch in range(output_no_channels):

                        # Get the image slice of pointed by the window coordinates
                        curr_img_padded_slice = img_padded[i, top:bottom, left:right, :]

                        # Calculate the derivative for the current image slice window
                        dimg_padded[i, top:bottom, left:right, :] += (
                            self.params[self.w_name][:, :, :, ch] * dprev[i, h, w, ch]
                        )

                        # Calculate derivative of the weight
                        dW[:, :, :, ch] += curr_img_padded_slice * dprev[i, h, w, ch]

                        # Calculate derivative of the bias
                        db[ch] += dprev[i, h, w, ch]

            # Set the dimg for current image without padding
            if self.padding > 0:
                dimg[i, :, :, :] = dimg_padded[i, self.padding : -self.padding, self.padding : -self.padding, :]
            else:
                dimg[i, :, :, :] = dimg_padded[i, :, :, :]

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

        # Iterate over each image in the batch
        for i in range(batch_size):

            # Start from the height
            for h in range(output_height):

                # Calulate the current window coordinates top and bottom
                top, bottom = h * self.stride, h * self.stride + self.pool_size

                # Towards the width
                # i.e. from top-to-bottom, left-to-right
                for w in range(output_width):

                    # Calulate the current window coordinates left and right
                    left, right = w * self.stride, w * self.stride + self.pool_size

                    # Loop over the channels i.e. the number of filters
                    for ch in range(out_channels):

                        # Get the part of the image acted upon by the current window
                        img_slice = img[i, top:bottom, left:right, ch]

                        # Do max pooling and store in the output
                        output[i, h, w, ch] = np.max(img_slice)
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
        create_window_max_mask = lambda x: x == np.max(x)

        # Get the batch size, and respective output dimensions
        batch_size = img.shape[0]
        _, h_out, w_out, channel_out = dprev.shape

        # Initialize dimg
        dimg = np.zeros(img.shape)

        # Itereate over all the image in the batch
        for i in range(batch_size):

            # Start from the height
            for h in range(h_out):

                # Calculate the current window coordinates top and bottom
                top, bottom = h * self.stride, h * self.stride + self.pool_size

                # Go towards the width
                # i.e. from top-to-bottom and left-to-right
                for w in range(w_out):

                    # Calculate the current window coordinates left anf right
                    left, right = w * self.stride, w * self.stride + self.pool_size

                    # Calculate for every channel in dprev i.e. the number of filters
                    for ch in range(channel_out):

                        # Get the current image part that is to be worked on
                        curr_img_slice = img[i, top:bottom, left:right, ch]

                        # Get the mask corresponding to the current image part
                        mask = create_window_max_mask(curr_img_slice)

                        # Calculate the derivative for current part
                        dimg[i, top:bottom, left:right, ch] += np.multiply(mask, dprev[i, h, w, ch])
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dimg
