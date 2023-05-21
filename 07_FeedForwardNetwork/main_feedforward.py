import sys
import time
import datetime
from functions_FFNN import *
sys.path.append('src')

import mnist_loader
import network2

current_time = datetime.datetime.now()
id_test = current_time.strftime("%Y-%m-%d_%H-%M-%S")

# Load MNIST data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Set default parameters for training all examples
epochs = 200
batch = 10
lr = 0.25
lmbda = 5.0


# -----------------------------------------------------------------------------
# # 1st network to train: 1 hidden layer with 30 neurons:
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: 1 Hidden Layer')
    print('Architecture: [784, 30, 10]')
    net_1HD = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)

    # Initial state: gradients and test accuracy
    norms_init = static_norms(training_data, net_1HD)
    test_accuracy_init = net_1HD.accuracy(test_data)
    print('Initial norms: ' + str(norms_init))

    # Gradients, validation cost and accuracy during training
    st = time.time()
    norms_train, validation_cost, validation_accuracy = training(
        training_data, validation_data, net_1HD,
        epochs=epochs, mini_batch_size=batch, eta=lr, lmbda=lmbda)
    elapsed_time = time.time() - st

    # Trained network: gradients and test accuracy
    norms_end = static_norms(training_data, net_1HD)
    test_accuracy_end = net_1HD.accuracy(test_data)
    # Include last gradient norm in the norms vector:
    norms_train.append(norms_end)

    params = [str(len(training_data)), str(len(validation_data)),
              str(len(test_data)),
              str(epochs), str(batch), str(lr), str(lmbda),
              'CrossEntropyCost', str(elapsed_time)]
    document_test('net_1HD', net_1HD.sizes,
                  test_accuracy_init, test_accuracy_end,
                  norms_train,
                  validation_accuracy, validation_cost,
                  params, id_test)


# -----------------------------------------------------------------------------
# # 2nd network to train: 1 hidden layer with 100 neurons:
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: 1 Hidden Layer')
    print('Architecture: [784, 100, 10]')
    net_1HD_100 = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)

    # Initial state: gradients and test accuracy
    norms_init = static_norms(training_data, net_1HD_100)
    test_accuracy_init = net_1HD_100.accuracy(test_data)
    print('Initial norms: ' + str(norms_init))

    # Gradients, validation cost and accuracy during training
    st = time.time()
    norms_train, validation_cost, validation_accuracy = training(
        training_data, validation_data, net_1HD_100,
        epochs=epochs, mini_batch_size=batch, eta=lr, lmbda=lmbda)
    elapsed_time = time.time() - st

    # Trained network: gradients and test accuracy
    norms_end = static_norms(training_data, net_1HD_100)
    test_accuracy_end = net_1HD_100.accuracy(test_data)
    # Include last gradient norm in the norms vector:
    norms_train.append(norms_end)

    params = [str(len(training_data)), str(len(validation_data)),
              str(len(test_data)),
              str(epochs), str(batch), str(lr), str(lmbda),
              'CrossEntropyCost', str(elapsed_time)]
    document_test('net_1HD_100', net_1HD_100.sizes,
                  test_accuracy_init, test_accuracy_end,
                  norms_train,
                  validation_accuracy, validation_cost,
                  params, id_test)


# -----------------------------------------------------------------------------
# # 3rd network to train: 2 hidden layers with 30 neurons:
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: 2 Hidden Layers')
    print('Architecture: [784, 30, 30, 10]')
    net_2HD = network2.Network([784, 30, 30, 10],
                               cost=network2.CrossEntropyCost)

    # Initial state: gradients and test accuracy
    norms_init = static_norms(training_data, net_2HD)
    test_accuracy_init = net_2HD.accuracy(test_data)
    print('Initial norms: ' + str(norms_init))

    # Gradients, validation cost and accuracy during training
    st = time.time()
    norms_train, validation_cost, validation_accuracy = training(
        training_data, validation_data, net_2HD,
        epochs=epochs, mini_batch_size=batch, eta=lr, lmbda=lmbda)
    elapsed_time = time.time() - st

    # Trained network: gradients and test accuracy
    norms_end = static_norms(training_data, net_2HD)
    test_accuracy_end = net_2HD.accuracy(test_data)
    # Include last gradient norm in the norms vector:
    norms_train.append(norms_end)

    params = [str(len(training_data)), str(len(validation_data)),
              str(len(test_data)),
              str(epochs), str(batch), str(lr), str(lmbda),
              'CrossEntropyCost', str(elapsed_time)]
    document_test('net_2HD', net_2HD.sizes,
                  test_accuracy_init, test_accuracy_end,
                  norms_train,
                  validation_accuracy, validation_cost,
                  params, id_test)


# -----------------------------------------------------------------------------
# # 4th network to train: 3 hidden layers with 30 neurons:
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: 3 Hidden Layers')
    print('Architecture: [784, 30, 30, 30, 10]')
    net_3HD = network2.Network([784, 30, 30, 30, 10],
                               cost=network2.CrossEntropyCost)

    # Initial state: gradients and test accuracy
    norms_init = static_norms(training_data, net_3HD)
    test_accuracy_init = net_3HD.accuracy(test_data)
    print('Initial norms: ' + str(norms_init))

    # Gradients, validation cost and accuracy during training
    st = time.time()
    norms_train, validation_cost, validation_accuracy = training(
        training_data, validation_data, net_3HD,
        epochs=epochs, mini_batch_size=batch, eta=lr, lmbda=lmbda)
    elapsed_time = time.time() - st

    # Trained network: gradients and test accuracy
    norms_end = static_norms(training_data, net_3HD)
    test_accuracy_end = net_3HD.accuracy(test_data)
    # Include last gradient norm in the norms vector:
    norms_train.append(norms_end)

    params = [str(len(training_data)), str(len(validation_data)),
              str(len(test_data)),
              str(epochs), str(batch), str(lr), str(lmbda),
              'CrossEntropyCost', str(elapsed_time)]
    document_test('net_3HD', net_3HD.sizes,
                  test_accuracy_init, test_accuracy_end,
                  norms_train,
                  validation_accuracy, validation_cost,
                  params, id_test)


# -----------------------------------------------------------------------------
# # 5th network to train: 5 hidden layers with 30 neurons:
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: 5 Hidden Layers')
    print('Architecture: [784, 30, 30, 30, 30, 30, 10]')
    net_5HD = network2.Network([784, 30, 30, 30, 30, 30, 10],
                               cost=network2.CrossEntropyCost)

    # Initial state: gradients and test accuracy
    norms_init = static_norms(training_data, net_5HD)
    test_accuracy_init = net_5HD.accuracy(test_data)
    print('Initial norms: ' + str(norms_init))

    # Gradients, validation cost and accuracy during training
    st = time.time()
    norms_train, validation_cost, validation_accuracy = training(
        training_data, validation_data, net_5HD,
        epochs=epochs, mini_batch_size=batch, eta=lr, lmbda=lmbda)
    elapsed_time = time.time() - st

    # Trained network: gradients and test accuracy
    norms_end = static_norms(training_data, net_5HD)
    test_accuracy_end = net_5HD.accuracy(test_data)
    # Include last gradient norm in the norms vector:
    norms_train.append(norms_end)

    params = [str(len(training_data)), str(len(validation_data)),
              str(len(test_data)),
              str(epochs), str(batch), str(lr), str(lmbda),
              'CrossEntropyCost', str(elapsed_time)]
    document_test('net_5HD', net_5HD.sizes,
                  test_accuracy_init, test_accuracy_end,
                  norms_train,
                  validation_accuracy, validation_cost,
                  params, id_test)


# -----------------------------------------------------------------------------
# # 6th network to train: 10 hidden layers with 30 neurons:
if True:
    # Initialize
    print('\n\n\n\n NEW CASE: 10 Hidden Layers')
    print('Architecture: [784, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 10]')
    net_10HD = network2.Network([784, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 10],
                               cost=network2.CrossEntropyCost)

    # Initial state: gradients and test accuracy
    norms_init = static_norms(training_data, net_10HD)
    test_accuracy_init = net_10HD.accuracy(test_data)
    print('Initial norms: ' + str(norms_init))

    # Gradients, validation cost and accuracy during training
    st = time.time()
    norms_train, validation_cost, validation_accuracy = training(
        training_data, validation_data, net_10HD,
        epochs=epochs, mini_batch_size=batch, eta=lr, lmbda=lmbda)
    elapsed_time = time.time() - st

    # Trained network: gradients and test accuracy
    norms_end = static_norms(training_data, net_10HD)
    test_accuracy_end = net_10HD.accuracy(test_data)
    # Include last gradient norm in the norms vector:
    norms_train.append(norms_end)

    params = [str(len(training_data)), str(len(validation_data)),
              str(len(test_data)),
              str(epochs), str(batch), str(lr), str(lmbda),
              'CrossEntropyCost', str(elapsed_time)]
    document_test('net_10HD', net_10HD.sizes,
                  test_accuracy_init, test_accuracy_end,
                  norms_train,
                  validation_accuracy, validation_cost,
                  params, id_test)
