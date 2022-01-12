from matplotlib import pyplot as plt
import numpy as np
import pickle
import math

def unpickle(file):
    with open(file, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    return data


def get_data(inputs_file_path):
    """
    Takes in an inputs file path and labels file path, loads the data into a dict,
    normalizes the inputs, and returns (NumPy array of inputs, NumPy
    array of labels).
    :param inputs_file_path: file path for ONE input batch, something like
    'cifar-10-batches-py/data_batch_1'
    :return: NumPy array of inputs as float32 and labels as int8
    """
    # TODO: Load inputs and labels
    input = unpickle(inputs_file_path)

    data = np.array(input[b"data"])
    labels = np.array(input[b"labels"])

    # TODO: Normalize inputs
    # Convert 0-255 into 0-1
    data = data / 255

    return (data, labels)


def softmax(x):
        """Calculates the softmax of a vector

        Args:
            x (ndarray[uint8]): An ndarray of uint8

        Returns:
            [ndarray[uint8]]: Vector after applying softmax
        """
        # Calculate exponentiation of the elements
        x = np.exp(x)

        # Calculate the sum of the elements
        sum_of_elements = np.sum(x)

        # Divide each element by the sum of the elements
        x = x / sum_of_elements

        # Return the vector (applied softmax)
        return x


def cross_entropy(y, p):
    """Calculates the cross entropy for the given vector

    Args:
        y (ndarray[unit8]): An ndarray vector
        p (ndarray[unit8]): An ndarray vector

    Returns:
        [float]: Cross entropy for two vectors
    """
    return -np.sum(y.T * np.log(p))


class Model:
    """
    This model class will contain the architecture for
    your single layer Neural Network for classifying CIFAR10 with
    batched learning. Please implement the TODOs for the entire
    model but do not change the method and constructor arguments.
    Make sure that your Model class works with multiple batch
    sizes. Additionally, please exclusively use NumPy and
    Python built-in functions for your implementation.
    """

    def __init__(self):
        # TODO: Initialize all hyperparametrs
        self.input_size = 3072  # Size of image vectors
        self.num_classes = 10  # Number of classes/possible labels
        self.batch_size = 100
        self.learning_rate = 0.005

        # TODO: Initialize weights and biases
        self.W = np.zeros((10, 3072))
        self.b = np.zeros((10,))

    def forward(self, inputs):
        """
        Does the forward pass on an batch of input images.
        :param inputs: normalized (0.0 to 1.0) batch of images,
                       (batch_size x 3072) (2D), where batch can be any number.
        :return: probabilities, probabilities for each class per image # (batch_size x 10)
        """
        # TODO: Write the forward pass logic for your model
        # TODO: Calculate, then return, the probability for each class per image using the Softmax equation
        batch_size, input_size = inputs.shape

        # Create an empty ndarray to store predictions for each image
        predictions = np.empty((batch_size, self.num_classes))

        # Calculate prediction for each input image and corresponding class
        for idx, x in enumerate(inputs):
            predictions[idx] = softmax(self.W * x + self.b)

        return predictions

    def loss(self, probabilities, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Loss should be generally decreasing with every training loop (step).
        :param probabilities: matrix that contains the probabilities
        of each class for each image
        :param labels: the true batch labels
        :return: average loss per batch element (float)
        """
        # TODO: calculate average cross entropy loss for a batch
        batch_size, num_of_classes = probabilities.shape

        # Calculate sum of cross entropies for each image in the batch
        sum_of_cross_entropies = 0
        for _ in range(batch_size):
            sum_of_cross_entropies += cross_entropy(labels, probabilities)

        # Calculate loss
        L = (1 / batch_size) * sum_of_cross_entropies

        return L

    def compute_gradients(self, inputs, probabilities, labels):
        """
        Returns the gradients for model's weights and biases
        after one forward pass and loss calculation. You should take the
        average of the gradients across all images in the batch.
        :param inputs: batch inputs (a batch of images)
        :param probabilities: matrix that contains the probabilities of each
        class for each image
        :param labels: true labels
        :return: gradient for weights,and gradient for biases
        """
        # TODO: calculate the gradients for the weights and the gradients for the bias with respect to average loss

        pass

    def accuracy(self, probabilities, labels):
        """
        Calculates the model's accuracy by comparing the number
        of correct predictions with the correct answers.
        :param probabilities: result of running model.forward() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        # TODO: calculate the batch accuracy
        batch_size = labels.shape

        accuracy = np.where(probabilities == labels, probabilities, labels) / batch_size
        
        return accuracy

    def gradient_descent(self, gradW, gradB):
        """
        Given the gradients for weights and biases, does gradient
        descent on the Model's parameters.
        :param gradW: gradient for weights
        :param gradB: gradient for biases
        :return: None
        """
        # TODO: change the weights and biases of the model to descent the gradient
        self.W = self.W - self.learning_rate * gradW
        self.b = self.b - self.learning_rate * gradB


def train(model, train_inputs, train_labels):
    """
    Trains the model on all of the inputs and labels.
    :param model: the initialized model to use for the forward
    pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training)
    :param train_inputs: train labels (all labels to use for training)
    :return: None
    """
    # TODO: Iterate over the training inputs and labels, in model.batch_size increments
    # TODO: For every batch, compute then descend the gradients for the model's weights
    # Optional TODO: Call visualize_loss and observe the loss per batch as the model trains.

    input_size, input_data_size = train_inputs.shape

    # Calculate number of batches
    num_of_batches = math.ceil(input_size / model.batch_size)

    # Train on all the batched
    for i in num_of_batches:
        # Create the batch
        current_idx = model.batch_size * i
        batch_inputs = train_inputs[current_idx : current_idx + model.batch_size]
        batch_labels = train_labels[current_idx: current_idx + model.batch_size]

        # Calculate probablities for the input belongning to a class
        probablibities = model.forward(batch_inputs)

        # Calculate loss and visualize it
        loss = model.loss(probablibities, batch_labels)
        visualize_loss(loss)

        # Calculate gradients
        gradW, gradB = model.compute_gradients(batch_inputs, probablibities, batch_labels)

        # Perform gradient descent
        model.gradient_descent(gradW, gradB)


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels.
    :param test_inputs: CIFAR10 test data (all images to be tested)
    :param test_labels: CIFAR10 test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """
    # TODO: Iterate over the testing inputs and labels
    # TODO: Return accuracy across testing set
    input_size, input_data_size = test_inputs.shape

    # Initialize probabilities
    probabilities = np.empty((input_size,))

    for idx, x in range(input_size):
        probabilities[idx] = np.amax(softmax(model.W * x + model.b), axis = 1)

    accuracy = model.accuracy(probabilities, test_labels)

    return accuracy


def hyperparameter_tuning(model, train_inputs, train_labels):
    """Hyperparameter

    Args:
        model ([Model]): The machine learning model to use
        train_inputs ([ndarray[uint8]]): All the training input
        train_labels ([ndarray[uint8]]): All the training labels

    Returns:
        batch_size, learning_rate ([tuple]): The optimal batch size and the learning rate
    """
    pass


def visualize_loss(losses):
    """
    Uses Matplotlib to visualize loss per batch. You can call this in train() to observe.
    param losses: an array of loss value from each batch of train

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """

    plt.ion()
    plt.show()

    x = np.arange(1, len(losses) + 1)
    plt.xlabel("i'th Batch")
    plt.ylabel("Loss Value")
    plt.title("Loss per Batch")
    plt.plot(x, losses, color="r")
    plt.draw()
    plt.pause(0.001)


def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data()
    :param probabilities: the output of model.forward()
    :param image_labels: the labels from get_data()

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """
    plt.ioff()

    images = np.reshape(image_inputs, (-1, 3, 32, 32))
    images = np.moveaxis(images, 1, -1)
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="both", which="both", length=0)
    plt.show()


def main():
    """
    Read in CIFAR10 data, initialize your model, and train and test your model
    for one epoch. The number of training steps should be your the number of
    batches you run through in a single epoch.
    :return: None
    """

    # TODO: load CIFAR10 train and test examples into train_inputs, train_labels, test_inputs, test_labels
    train_inputs, train_labels = get_data('./data/cifar-10-batches-py/data_batch_1')
    test_inputs, test_labels = get_data('./data/cifar-10-batches-py/test_batch')

    # TODO: Create Model
    model = Model()

    # TODO: Train model by calling train() ONCE on all data
    train(model, train_inputs, train_labels)

    # TODO: Test the accuracy by calling test() after running train()
    accuracy = test(model, test_inputs, test_labels)
    print(f'{accuracy:.4f}')

    # TODO: Visualize the data by using visualize_results() on a set of 10 examples
    visualize_results(test_inputs, model.forward(), test_labels)


if __name__ == "__main__":
    main()
