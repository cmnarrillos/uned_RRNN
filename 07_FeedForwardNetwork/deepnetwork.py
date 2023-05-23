import sys
import time
import datetime
sys.path.append('src')

# Import modules from M.Nielsen code at ./src/
import network4
from network4 import Network
from network4 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer


# Identify the test (for saving results)
current_time = datetime.datetime.now()
id_test = current_time.strftime("%Y-%m-%d_%H-%M-%S")+'_deep'

# Load MNIST data
training_data, validation_data, test_data = network4.load_data_shared()

# Set default parameters for training all examples
epochs = 6
mini_batch_size = 10
lr = 0.5
lmbda = 5.0


# -----------------------------------------------------------------------------
# # 1st network to train: 1 hidden layer with 100 neurons:
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: 1 Hidden Layer')
    print('Architecture: [784, 100, 10]')
    net = Network([
                    FullyConnectedLayer(n_in=784, n_out=100),
                    SoftmaxLayer(n_in=100, n_out=10)
                    ], mini_batch_size)

    # Train the network
    net.SGD(training_data, epochs, mini_batch_size, lr,
                validation_data, test_data, lmbda=lmbda)
