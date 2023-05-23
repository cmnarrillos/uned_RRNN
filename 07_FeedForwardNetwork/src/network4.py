"""network3.py
~~~~~~~~~~~~~~

A Theano-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.

Because the code is based on Theano, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.

This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).

"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np
import tensorflow as tf

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return tf.maximum(0.0, z)
def ReLU_mod(z): return tf.maximum(z/1000, z)
def sigmoid(z): return tf.sigmoid(z)
def tanh(z): return tf.tanh(z)

#### Constants
GPU = True
if GPU:
    print("Trying to run under a GPU. If this is not desired, then modify " +
          "network3.py\n to set the GPU flag to False.")
    try:
        tf.config.experimental.set_memory_growth(
            tf.config.experimental.list_physical_devices('GPU')[0], True)
    except:
        pass
else:
    print("Running with a CPU. If this is not desired, then modify " +
          "network3.py to set\n the GPU flag to True.")

#### Load the MNIST data
def load_data_shared(filename="./data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    def shared(data):
        """Place the data into shared variables. This allows TensorFlow to access the data efficiently."""
        shared_x = tf.Variable(
            np.asarray(data[0], dtype=np.float32), trainable=False)
        shared_y = tf.Variable(
            np.asarray(data[1], dtype=np.int32), trainable=False)
        return shared_x, tf.cast(shared_y, tf.int32)
    return [shared(training_data), shared(validation_data), shared(test_data)]

#### Main class used to construct and train networks
class Network(object):
    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent."""
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = tf.placeholder(tf.float32, [None, layers[0].n_in])
        self.y = tf.placeholder(tf.int32, [None])
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j - 1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation, and testing
        num_training_batches = int(len(training_data[0]) / mini_batch_size)
        num_validation_batches = int(len(validation_data[0]) / mini_batch_size)
        num_test_batches = int(len(test_data[0]) / mini_batch_size)

        # define the (regularized) cost function, gradients, and updates
        l2_norm_squared = sum([tf.reduce_sum(tf.square(layer.w)) for layer in self.layers])
        cost = self.layers[-1].cost(self) + 0.5 * lmbda * l2_norm_squared / num_training_batches
        grads = tf.gradients(cost, self.params)
        updates = [(param, param - eta * grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = tf.placeholder(tf.int32)  # mini-batch index
        train_mb = tf.function(
            [i], cost, updates=updates,
            givens={
                self.x: training_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y: training_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })
        validate_mb_accuracy = tf.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x: validation_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y: validation_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })
        test_mb_accuracy = tf.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x: test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y: test_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })
        self.test_mb_predictions = tf.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x: test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })
        # Do the actual training
        best_validation_accuracy = 0.0
        for epoch in range(epochs):
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches * epoch + minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                if (iteration + 1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in range(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in range(num_test_batches)])
                            print('The corresponding test accuracy is {0:.2%}'.format(
                                test_accuracy))
        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

#### Define layer types

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer. A more sophisticated implementation would separate the
    two, but for our purposes, we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.
    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=tf.sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.
        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
        self.w = tf.Variable(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0 / n_out), size=filter_shape),
                dtype=tf.float32),
            name='w')
        self.b = tf.Variable(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=tf.float32),
            name='b')
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = tf.reshape(inpt, self.image_shape)
        conv_out = tf.nn.conv2d(
            input=self.inpt, filters=self.w, strides=[1, 1, 1, 1], padding='SAME')
        pooled_out = tf.nn.max_pool2d(
            input=conv_out, ksize=[1, self.poolsize[0], self.poolsize[1], 1],
            strides=[1, self.poolsize[0], self.poolsize[1], 1], padding='VALID')
        self.output = self.activation_fn(pooled_out + tf.reshape(self.b, [1, -1, 1, 1]))
        self.output_dropout = self.output  # no dropout in the convolutional layers

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=tf.sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = tf.Variable(
            np.asarray(
                np.random.normal(loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)),
                dtype=tf.float32),
            name='w')
        self.b = tf.Variable(
            np.asarray(
                np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                dtype=tf.float32),
            name='b')
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = tf.reshape(inpt, (mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1 - self.p_dropout) * tf.matmul(self.inpt, self.w) + self.b)
        self.y_out = tf.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            tf.reshape(inpt_dropout, (mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            tf.matmul(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return tf.reduce_mean(tf.cast(tf.equal(y, self.y_out), tf.float32))

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = tf.Variable(
            np.zeros((n_in, n_out), dtype=tf.float32),
            name='w')
        self.b = tf.Variable(
            np.zeros((n_out,), dtype=tf.float32),
            name='b')
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = tf.reshape(inpt, (mini_batch_size, self.n_in))
        self.output = tf.nn.softmax((1 - self.p_dropout) * tf.matmul(self.inpt, self.w) + self.b)
        self.y_out = tf.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            tf.reshape(inpt_dropout, (mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = tf.nn.softmax(tf.matmul(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        return -tf.reduce_mean(tf.log(self.output_dropout) * tf.one_hot(net.y, self.n_out))

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return tf.reduce_mean(tf.cast(tf.equal(y, self.y_out), tf.float32))


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].shape[0]

def dropout_layer(layer, p_dropout):
    mask = tf.cast(tf.random.uniform(layer.shape) > p_dropout, tf.float32)
    return layer * mask
