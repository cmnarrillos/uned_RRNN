import math
import os
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce


def training(tr_d, val_d, net, epochs, mini_batch_size=1000, eta=0.1, lmbda=5.0):
    norms = []
    validation_cost = []
    validation_accuracy = []
    for j in range(epochs):
        average_gradient = get_average_gradient(net, tr_d)
        norms.append([list_norm(avg) for avg in average_gradient[:-1]])
        print("Epoch: %s" % j)
        val_cost, val_acc, tr_cost, tr_acc = \
                            net.SGD(tr_d, 1, mini_batch_size, eta,
                                    lmbda=lmbda,
                                    evaluation_data=val_d,
                                    monitor_evaluation_cost=True,
                                    monitor_evaluation_accuracy=True)
        validation_cost.append(val_cost)
        validation_accuracy.append(val_acc)
    return norms, validation_cost, validation_accuracy


def static_norms(training_data, net):
    average_gradient = get_average_gradient(net, training_data)
    norms = [list_norm(avg) for avg in average_gradient[:-1]]
    return norms


def get_average_gradient(net, training_data):
    nabla_b_results = [net.backprop(x, y)[0] for x, y in training_data]
    gradient = list_sum(nabla_b_results)
    return [(np.reshape(g, len(g))/len(training_data)).tolist()
            for g in gradient]


def zip_sum(a, b):
    return [x + y for (x, y) in zip(a, b)]


def list_sum(l):
    return reduce(zip_sum, list(l))


def list_norm(l):
    return math.sqrt(sum([x*x for x in l]))


def document_test(filename, network_sizes, test_acc_init, test_acc_end,
                  norms, validation_acc_train, validation_cost_train,
                  params, id_test):
    if not os.path.exists('./tests/' + id_test):
        os.makedirs('./tests/' + id_test)

    plot_grad_evolution(norms, './tests/' + id_test + '/' + filename +
                        '_grad.png')
    plot_grad_evolution(norms, './tests/' + id_test + '/' + filename +
                        '_gradlog.png', log=True)
    plot_val_evolution(validation_acc_train, './tests/' + id_test + '/' +
                       filename + '_accuracy.png', int(params[1]),
                       accuracy=True)
    plot_val_evolution(validation_cost_train, './tests/' + id_test + '/' +
                       filename + '_cost.png', accuracy=False)

    with open('./tests/' + id_test + '/' + filename + '.txt', 'w') as f:
        f.write('Network architecture:\n')
        f.write(str(network_sizes) + '\n')
        f.write('\n')
        f.write('Training params:\n')
        f.write(' - Training set size: ' + params[0] + '\n')
        f.write(' - Validation set size: ' + params[1] + '\n')
        f.write(' - Test set size: ' + params[2] + '\n')
        f.write(' - Number of epoch: ' + params[3] + '\n')
        f.write(' - Mini batch size: ' + params[4] + '\n')
        f.write(' - Learning rate (eta): ' + params[5] + '\n')
        f.write(' - Regularization factor (lambda): ' + params[6] + '\n')
        f.write(' - Cost function: ' + params[7] + '\n')
        f.write(' - Elapsed time: ' + params[8] + 's\n')
        f.write('\n')
        f.write('Test set accuracy:\n')
        f.write(' - Initial state: ' + str(test_acc_init) + '/' + params[2]+'\n')
        f.write(' - After training: ' + str(test_acc_end) + '/' + params[2]+'\n')
        f.write('\n')
        f.write('Validation set accuracy during training:\n')
        for ii, validation_acc in enumerate(validation_acc_train):
            f.write(' - Epoch ' + str(ii) + ': ' + str(validation_acc)
                    + '/' + params[1] + '\n')
        f.write('\n')
        f.write('Gradient vector norms:\n')
        f.write(' - Initial: ' + str(norms[0]) + '\n')
        for ii, norm in enumerate(norms[1:-1]):
            f.write(' - Epoch ' + str(ii) + ': ' + str(norm) + '\n')
        f.write(' - Final:   ' + str(norms[-1]) + '\n')


def plot_grad_evolution(gradient, filename, log=False):
    epochs = range(1, len(gradient) + 1)
    gradient = np.array(gradient)

    fig, ax = plt.subplots()

    for i in range(gradient.shape[1]):
        if log:
            ax.semilogy(epochs, gradient[:, i], label=f'Layer {i+1}')
        else:
            ax.plot(epochs, gradient[:, i], label=f'Layer {i+1}')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Gradient Norm')
    ax.set_title('Gradient Evolution')
    ax.legend()
    ax.grid()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    if len(gradient) > 20:
        fig, ax = plt.subplots()

        for i in range(gradient.shape[1]):
            if log:
                ax.semilogy(epochs[:20], gradient[:20, i], label=f'Layer {i+1}')
            else:
                ax.plot(epochs[:20], gradient[:20, i], label=f'Layer {i+1}')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Gradient Norm')
        ax.set_title('Gradient Evolution')
        ax.legend()
        ax.grid()

        plt.tight_layout()
        plt.savefig(filename[:-4] + '_init.png')
        plt.close()


def plot_val_evolution(val_metric, filename,
                       set_size=10000, accuracy=False):
    epochs = range(1, len(val_metric) + 1)
    if accuracy:
        val_metric = np.array(val_metric)*(100/set_size)
        label = 'Validation Accuracy (%)'
    else:
        val_metric = np.array(val_metric)
        label = 'Validation Cost'

    fig, ax = plt.subplots()

    ax.plot(epochs, val_metric)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(label)
    ax.set_title('Validation Evolution')
    ax.grid()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    if len(val_metric) > 20:
        fig, ax = plt.subplots()

        ax.plot(epochs[:20], val_metric[:20])

        ax.set_xlabel('Epoch')
        ax.set_ylabel(label)
        ax.set_title('Validation Evolution')
        ax.grid()

        plt.tight_layout()
        plt.savefig(filename[:-4] + '_init.png')
        plt.close()
