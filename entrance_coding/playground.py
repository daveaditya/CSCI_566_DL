from matplotlib import pyplot as plt
import numpy as np
import pickle
import math
import itertools
from timeit import default_timer as timer

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
        # For numerical stability
        exp = np.exp(x - np.max(x))

        # Calculate exponentiation of the elements
        for i in range(len(x)):
            exp[i] /= np.sum(exp[i])

        # Return the vector (applied softmax)
        return exp


def cross_entropy(y, p):
    """Calculates the cross entropy for the given vector

    Args:
        y (ndarray[unit8]): True labels
        p (ndarray[unit8]): Probabilities

    Returns:
        [float]: Cross entropy for two vectors
    """
    return - np.log(p[np.arange(len(y)), y])


def one_hot_encode(labels, num_of_classes):
    """One hot encodes the labels vector

    Args:
        labels ([ndarray]): A vector of indices
        num_of_classes ([int]): Number of classes

    Returns:
        [ndarray[uint8]]: A matrix of labels.size x num_of_classes
    """
    return np.identity(num_of_classes)[labels]


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
        self.batch_size = 16
        self.learning_rate = 0.003

        # TODO: Initialize weights and biases
        self.W = np.zeros((3072, 10))
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
        # Calculate prediction for each input image and corresponding class
        probabilities = softmax(np.dot(inputs, self.W) + self.b)

        return probabilities

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
        # Calculate sum of cross entropies for each image in the batch
        sum_of_cross_entropies = 0
        for _ in range(self.batch_size):
            sum_of_cross_entropies += cross_entropy(labels, probabilities)

        # Calculate loss
        L = (1 / self.batch_size) * sum_of_cross_entropies

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
        num_examples = labels.size

        # create one hot encoded vector
        one_hot_encoded_labels = one_hot_encode(labels, self.num_classes)

        gradW = (1 / num_examples) * np.dot(inputs.T, (probabilities - one_hot_encoded_labels))
        gradB = (1 / num_examples) * np.sum(probabilities - one_hot_encoded_labels)

        return (gradW, gradB)

    def accuracy(self, probabilities, labels):
        """
        Calculates the model's accuracy by comparing the number
        of correct predictions with the correct answers.
        :param probabilities: result of running model.forward() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        # TODO: calculate the batch accuracy
        num_examples = labels.size

        accuracy = np.sum(probabilities == labels) / num_examples

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

    loss = list()

    # Train on all the batched
    for i in range(num_of_batches):
        # Create the batch
        current_idx = model.batch_size * i
        batch_inputs = train_inputs[current_idx : current_idx + model.batch_size]
        batch_labels = train_labels[current_idx: current_idx + model.batch_size]

        # Calculate probablities for the input belongning to a class
        probabilities = model.forward(batch_inputs)

        # Calculate loss and visualize it
        loss.append(model.loss(probabilities, batch_labels))

        # Calculate gradients
        gradW, gradB = model.compute_gradients(batch_inputs, probabilities, batch_labels)

        # Perform gradient descent
        model.gradient_descent(gradW, gradB)

    visualize_loss(loss)


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels.
    :param test_inputs: CIFAR10 test data (all images to be tested)
    :param test_labels: CIFAR10 test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """
    # TODO: Iterate over the testing inputs and labels
    # TODO: Return accuracy across testing set
    probabilities = np.argmax(softmax(np.dot(test_inputs, model.W) + model.b), axis = 1)

    accuracy = model.accuracy(probabilities, test_labels)

    return accuracy


def hyperparameter_tuning(inputs, labels, test_inputs, test_labels):
    """Hyperparameter

    Args:
        model ([Model]): The machine learning model to use
        train_inputs ([ndarray[uint8]]): All the training input
        train_labels ([ndarray[uint8]]): All the training labels

    Returns:
        batch_size, learning_rate ([tuple]): The optimal batch size and the learning rate
    """

    batch_sizes = [16]
    learning_rates = [0.003, 0.0035, 0.004, 0.0045, 0.005]
    hyperparameters = itertools.product(batch_sizes, learning_rates)
    results = list()

    for batch_size, learning_rate in hyperparameters:
        # Split data into train and validation
        mask = np.random.rand(len(inputs)) <= 0.7
        train_inputs, train_labels = inputs[mask], labels[mask]
        validation_inputs, validation_labels = inputs[~mask], labels[~mask]

        model = Model()

        # Set model parameters
        model.batch_size = batch_size
        model.learning_rate = learning_rate

        start = timer()
        train(model, train_inputs, train_labels)

        validation_accuracy = test(model, validation_inputs, validation_labels)
        test_accuracy = test(model, test_inputs, test_labels)

        end = timer()

        result = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'val_accuracy': validation_accuracy,
            'test_accuracy': test_accuracy,
            'total_time_in_seconds': end - start
        }
        print(result)
        results.append(result)

        del model

    return results


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
    # Get all train inputs and labels
    input_file_paths = [
        './data/cifar-10-batches-py/data_batch_1',
        './data/cifar-10-batches-py/data_batch_2',
        './data/cifar-10-batches-py/data_batch_3',
        './data/cifar-10-batches-py/data_batch_4',
        './data/cifar-10-batches-py/data_batch_5'
    ]

    list_train_inputs, list_train_labels = (list(), list())
    for input_file_path in input_file_paths:
        curr_train_inputs, curr_train_labels = get_data(input_file_path)
        list_train_inputs.append(curr_train_inputs)
        list_train_labels.append(curr_train_labels)


    # TODO: load CIFAR10 train and test examples into train_inputs, train_labels, test_inputs, test_labels
    train_inputs = np.concatenate(list_train_inputs, axis = 0)
    train_labels = np.concatenate(list_train_labels, axis = 0)

    test_inputs, test_labels = get_data('./data/cifar-10-batches-py/test_batch')

    # # TODO: Create Model
    model = Model()

    # Used for fine tuning hyperparameters
    # hyperparameter_tuning(train_inputs, train_labels, test_inputs, test_labels)

    # TODO: Train model by calling train() ONCE on all data
    train(model, train_inputs, train_labels)

    # TODO: Test the accuracy by calling test() after running train()
    accuracy = test(model, test_inputs, test_labels)
    print(f'{accuracy:.4f}')

    # TODO: Visualize the data by using visualize_results() on a set of 10 examples
    mask = np.random.choice(len(test_inputs), 10)
    selected_inputs = test_inputs[mask]
    selected_labels = test_labels[mask]
    visualize_results(selected_inputs, model.forward(selected_inputs), selected_labels)


if __name__ == "__main__":
    main()
