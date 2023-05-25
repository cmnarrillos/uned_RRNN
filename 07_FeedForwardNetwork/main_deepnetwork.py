import sys
import time
import datetime
sys.path.append('src')

# Import modules from M.Nielsen code at ./src/
import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from network3 import ReLU, ReLU_mod


# Identify the test (for saving results)
current_time = datetime.datetime.now()
id_test = current_time.strftime("%Y-%m-%d_%H-%M-%S")+'_deep'

# Load MNIST data
training_data, validation_data, test_data = network3.load_data_shared()
expanded_training_data, _, _ = network3.load_data_shared(
    './data/mnist_expanded.pkl.gz')

# Set default parameters for training all examples
epochs = 6
mini_batch_size = 10
lr = 0.5
lmbda = 5.0
dropout = 0.2


# -----------------------------------------------------------------------------
# # 1st network to train: 1 hidden layer with 100 neurons:
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: 1 FullyConnected Layer')
    print('Architecture: [784, 100, 10]')
    net = Network([
                    FullyConnectedLayer(n_in=784, n_out=100),
                    SoftmaxLayer(n_in=100, n_out=10)
                    ], mini_batch_size)

    # Train the network
    net.SGD(training_data, epochs, mini_batch_size, lr,
                validation_data, test_data, lmbda=lmbda)

# -----------------------------------------------------------------------------
# # 2nd network to train: 1 conv-pool + 1 FC layer
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + FC Layer')
    print('Architecture: [784, 20x(24,24), 100, 10]')
    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=20 * 12 * 12, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

    # Train the network
    net.SGD(training_data, epochs, mini_batch_size, lr,
                validation_data, test_data, lmbda=lmbda)

# -----------------------------------------------------------------------------
# # 3rd network to train: 2 conv-pool + 1 FC layer
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
                             'Convolutional + Pool + FC Layer')
    print('Architecture: [784, 20x(24,24), 20x(12,12), 100, 10]')
    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=40*4*4, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

    # Train the network
    net.SGD(training_data, epochs, mini_batch_size, lr,
                validation_data, test_data, lmbda=lmbda)

# -----------------------------------------------------------------------------
# # 4th network to train: 2 conv-pool + 1 FC layer with ReLU
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
                             'Convolutional + Pool + FC Layer (ReLU)')
    print('Architecture: [784, 20x(24,24), 20x(12,12), 100, 10]')
    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

    # Train the network
    net.SGD(training_data, epochs, mini_batch_size, lr,
                validation_data, test_data, lmbda=lmbda)

# -----------------------------------------------------------------------------
# # 5th network to train: 2 conv-pool + 1 FC layer with modified ReLU
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
                             'Convolutional + Pool + FC Layer(ReLU_mod)')
    print('Architecture: [784, 20x(24,24), 20x(12,12), 100, 10]')
    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU_mod),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

    # Train the network
    net.SGD(training_data, epochs, mini_batch_size, lr,
                validation_data, test_data, lmbda=lmbda)


# -----------------------------------------------------------------------------
# # 6th network to train: 2 conv-pool + 1 FC layer with modified ReLU
# # expanding training data to 250.000
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
                             'Convolutional + Pool + FC Layer')
    print('Architecture: [784, 20x(24,24), 20x(12,12), 100, 10]')
    print('Expanded training data')
    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU_mod),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

    # Train the network
    net.SGD(expanded_training_data, epochs, mini_batch_size, lr,
                validation_data, test_data, lmbda=lmbda)


# -----------------------------------------------------------------------------
# # 7.1th network to train: 2 conv-pool + 2 FC layers with sigmoid
# # expanding training data to 250.000
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
                             'Convolutional + Pool + 2 FC Layers')
    print('Architecture: [784, 20x(24,24), 20x(12,12), 100, 100, 10]')
    print('Expanded training data')
    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=40*4*4, n_out=100),
        FullyConnectedLayer(n_in=100, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

    # Train the network
    net.SGD(expanded_training_data, epochs, mini_batch_size, lr,
                validation_data, test_data, lmbda=lmbda)


# -----------------------------------------------------------------------------
# # 7.2th network to train: 2 conv-pool + 2 FC layers with modified ReLU
# # expanding training data to 250.000
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
                             'Convolutional + Pool + 2 FC Layers')
    print('Architecture: [784, 20x(24,24), 20x(12,12), 100, 100, 10]')
    print('Expanded training data')
    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
        FullyConnectedLayer(n_in=100, n_out=100, activation_fn=ReLU),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

    # Train the network
    net.SGD(expanded_training_data, epochs, mini_batch_size, lr,
                validation_data, test_data, lmbda=lmbda)


# -----------------------------------------------------------------------------
# # 7.3th network to train: 2 conv-pool + 2 FC layers with modified ReLU
# # expanding training data to 250.000
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
                             'Convolutional + Pool + 2 FC Layers')
    print('Architecture: [784, 20x(24,24), 20x(12,12), 100, 100, 10]')
    print('Expanded training data')
    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU_mod),
        FullyConnectedLayer(n_in=100, n_out=100, activation_fn=ReLU_mod),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

    # Train the network
    net.SGD(expanded_training_data, epochs, mini_batch_size, lr,
                validation_data, test_data, lmbda=lmbda)


# -----------------------------------------------------------------------------
# # 8.1th network to train: 2 conv-pool + 2 FC layers with sigmoid
# # expanding training data to 250.000. Include dropout
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
                             'Convolutional + Pool + 2FC Layers')
    print('Architecture: [784, 20x(24,24), 20x(12,12), 100, 100, 10]')
    print('Expanded training data')
    print('Include dropout at FC layers')
    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=40*4*4, n_out=100, p_dropout=dropout),
        FullyConnectedLayer(n_in=100, n_out=100, p_dropout=dropout),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

    # Train the network
    net.SGD(expanded_training_data, epochs, mini_batch_size, lr,
                validation_data, test_data, lmbda=lmbda)



# -----------------------------------------------------------------------------
# # 8.2th network to train: 2 conv-pool + 2 FC layers with ReLU
# # expanding training data to 250.000. Include dropout
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
                             'Convolutional + Pool + 2FC Layers')
    print('Architecture: [784, 20x(24,24), 20x(12,12), 100, 100, 10]')
    print('Expanded training data')
    print('Include dropout at FC layers')
    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=40*4*4, n_out=100,
                            activation_fn=ReLU, p_dropout=dropout),
        FullyConnectedLayer(n_in=100, n_out=100,
                            activation_fn=ReLU, p_dropout=dropout),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

    # Train the network
    net.SGD(expanded_training_data, epochs, mini_batch_size, lr,
                validation_data, test_data, lmbda=lmbda)



# -----------------------------------------------------------------------------
# # 8.3th network to train: 2 conv-pool + 2 FC layers with modified ReLU
# # expanding training data to 250.000. Include dropout
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
                             'Convolutional + Pool + 2FC Layers')
    print('Architecture: [784, 20x(24,24), 20x(12,12), 100, 100, 10]')
    print('Expanded training data')
    print('Include dropout at FC layers')
    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=40*4*4, n_out=100,
                            activation_fn=ReLU_mod, p_dropout=dropout),
        FullyConnectedLayer(n_in=100, n_out=100,
                            activation_fn=ReLU_mod, p_dropout=dropout),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

    # Train the network
    net.SGD(expanded_training_data, epochs, mini_batch_size, lr,
                validation_data, test_data, lmbda=lmbda)



# -----------------------------------------------------------------------------
# # 9th network to train: 2 conv-pool + 3 FC layers with modified ReLU
# # expanding training data to 250.000. Include dropout
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
                             'Convolutional + Pool + 3FC Layers')
    print('Architecture: [784, 20x(24,24), 20x(12,12), 100, 100, 100, 10]')
    print('Expanded training data')
    print('Include dropout at FC layers')
    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=40*4*4, n_out=100,
                            activation_fn=ReLU_mod, p_dropout=dropout),
        FullyConnectedLayer(n_in=100, n_out=100,
                            activation_fn=ReLU_mod, p_dropout=dropout),
        FullyConnectedLayer(n_in=100, n_out=100,
                            activation_fn=ReLU_mod, p_dropout=dropout),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

    # Train the network
    net.SGD(expanded_training_data, epochs, mini_batch_size, lr,
                validation_data, test_data, lmbda=lmbda)



# -----------------------------------------------------------------------------
# # 10th network to train: 2 conv-pool + 4 FC layers with modified ReLU
# # expanding training data to 250.000. Include dropout
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
                             'Convolutional + Pool + 4FC Layers')
    print('Architecture: [784, 20x(24,24), 20x(12,12), '
                         '100, 100, 100, 100, 10]')
    print('Expanded training data')
    print('Include dropout at FC layers')
    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=40*4*4, n_out=100,
                            activation_fn=ReLU_mod, p_dropout=dropout),
        FullyConnectedLayer(n_in=100, n_out=100,
                            activation_fn=ReLU_mod, p_dropout=dropout),
        FullyConnectedLayer(n_in=100, n_out=100,
                            activation_fn=ReLU_mod, p_dropout=dropout),
        FullyConnectedLayer(n_in=100, n_out=100,
                            activation_fn=ReLU_mod, p_dropout=dropout),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

    # Train the network
    net.SGD(expanded_training_data, epochs, mini_batch_size, lr,
                validation_data, test_data, lmbda=lmbda)



# -----------------------------------------------------------------------------
# # 11th network to train: 2 conv-pool + 5 FC layers with modified ReLU
# # expanding training data to 250.000. Include dropout
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: Convolutional + Pool + '
                             'Convolutional + Pool + 5FC Layers')
    print('Architecture: [784, 20x(24,24), 20x(12,12), '
                         '100, 100, 100, 100, 100, 10]')
    print('Expanded training data')
    print('Include dropout at FC layers')
    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=40*4*4, n_out=100,
                            activation_fn=ReLU_mod, p_dropout=dropout),
        FullyConnectedLayer(n_in=100, n_out=100,
                            activation_fn=ReLU_mod, p_dropout=dropout),
        FullyConnectedLayer(n_in=100, n_out=100,
                            activation_fn=ReLU_mod, p_dropout=dropout),
        FullyConnectedLayer(n_in=100, n_out=100,
                            activation_fn=ReLU_mod, p_dropout=dropout),
        FullyConnectedLayer(n_in=100, n_out=100,
                            activation_fn=ReLU_mod, p_dropout=dropout),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

    # Train the network
    net.SGD(expanded_training_data, epochs, mini_batch_size, lr,
                validation_data, test_data, lmbda=lmbda)
