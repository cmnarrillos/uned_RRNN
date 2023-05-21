import numpy as np
import matplotlib.pyplot as plt
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List


def matrix_to_vector(matrix: np.ndarray) -> np.ndarray:
    """
    Convert a square matrix into a vector by concatenating the rows.

    Args:
        matrix: A square matrix.

    Returns:
        A 1D numpy array representing the vector obtained by concatenating the
        rows of the matrix.
    """
    return np.concatenate(matrix)


def vector_to_matrix(vector: np.ndarray) -> np.ndarray:
    """
    Convert a vector into the original square matrix.

    Args:
        vector: A 1D numpy array representing the vector obtained by
        concatenating the rows of the matrix.

    Returns:
        A square matrix.
    """
    n = int(np.sqrt(len(vector)))
    return np.reshape(vector, (n, n))


def plot_matrix(matrix: np.ndarray, filename: str) -> None:
    """
    Plot a matrix as a black and white image.

    Args:
        matrix: A 2D numpy array representing the matrix to plot.
        filename: name of the file where the pattern is stored

    Returns:
        None
    """
    plt.imshow(np.ones(np.shape(matrix))-matrix, cmap='gray', vmin=0, vmax=2)
    plt.savefig(filename, format='png')


def interpolate_pattern(filename: str, n: int) -> np.ndarray:
    """
    Interpolates a 2D binary pattern read from a file to a nxn 2D array using
    nearest-neighbor interpolation.

    Parameters:
        filename (str): The name of the file containing the input pattern. The
            file should contain a rectangular grid of 0s and 1s, where 0
            represents a black pixel and 1 represents a white pixel. Any other
            character will be interpreted as a space. Each row of the grid
            should be terminated with a newline character.
        n (int): The number of rows and columns in the interpolated pattern.

    Returns:
        np.ndarray: A 2D numpy array of shape (n, n) containing the interpolated
            pattern.

    Raises:
        ValueError: If the input pattern is not rectangular or contains invalid
            characters.
    """
    # read the pattern from the file
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines if line.strip() != '']
    # determine the number of columns in the pattern
    cols = max([len(line) for line in lines])

    # pad the lines with spaces to make them the same length
    lines = [line.ljust(cols, ' ') for line in lines]

    # convert the lines into a 2D numpy array of 0s and 1s
    pattern = np.array([[1 if c == '*' else -1 if c == ' ' else int(c)
                         for c in line] for line in lines])

    # get the size of the original pattern
    rows = pattern.shape[0]

    # create a matrix with the target size and fill it with zeros
    interpolated = np.zeros((n, n))

    # compute the indices of the rows and columns to copy the pattern to
    row_indices = np.arange(0, n).astype(int)
    col_indices = np.arange(0, n).astype(int)

    # copy the pattern into the center of the target matrix
    row_factor = n/rows
    col_factor = n/cols
    row_indices_orig = np.floor(row_indices/row_factor).astype(int)
    col_indices_orig = np.floor(col_indices/col_factor).astype(int)
    interpolated[np.ix_(row_indices, col_indices)] = \
        pattern[np.ix_(row_indices_orig, col_indices_orig)]

    return interpolated


def write_pattern_to_file(matrix: np.ndarray, filename: str) -> None:
    """
    Writes a 2D binary pattern to a file as a rectangular grid of asterisks (*)
    and spaces.

    Parameters:
        matrix (np.ndarray): A 2D numpy array containing the binary pattern to
            write. Each element of the array should be either 1 or -1,
            representing white and black pixels, respectively.
        filename (str): The name of the file to write the pattern to. If the file
            exists, its contents will be overwritten.

    Returns:
        None: This function does not return anything.
    """
    with open(filename, 'w') as f:
        for row in matrix:
            line = ''.join(['*' if x == 1 else ' ' for x in row])
            f.write(line + '\n')


def hopfield_weights(patterns: np.ndarray, d: float) -> np.ndarray:
    """
    Computes the weights of a Hopfield network given a set of binary patterns and
    a dilution parameter.

    The weights are computed using the Hopfield model, which is a form of auto-
    associative memory that can store and recall binary patterns. The dilution
    parameter d determines the proportion of the weight matrix that is set to zero.

    Parameters:
        patterns (np.ndarray): A 2D numpy array containing the binary patterns to
            store in the network. Each row represents a pattern, and each element
            of the row should be either 1 or -1, representing white and black
            pixels, respectively.
        d (float): The dilution parameter of the weight matrix. Must be a value
            between 0 and 1, inclusive. A value of 0 means that the weight matrix
            is fully connected, while a value of 1 means that all weights are zero.

    Returns:
        np.ndarray: A 2D numpy array containing the weights of the Hopfield
            network. The element w[i,j] represents the weight between neurons
            i and j in the network.
    """
    # Define weights according to hopfield model for multiple patterns
    n = patterns.shape[0]
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if random.random() < d:
                w[i, j] = 0
            else:
                w[i, j] = 1 / n * np.sum(patterns[i] * patterns[j])
    return w


def hopfield_weights_parallel(patterns: np.ndarray, d: float) -> np.ndarray:
    """
    Computes the weights of a Hopfield network given a set of binary patterns and
    a dilution parameter.

    The weights are computed using the Hopfield model, which is a form of auto-
    associative memory that can store and recall binary patterns. The dilution
    parameter d determines the proportion of the weight matrix that is set to zero.

    Parameters:
        patterns (np.ndarray): A 2D numpy array containing the binary patterns to
            store in the network. Each row represents a pattern, and each element
            of the row should be either 1 or -1, representing white and black
            pixels, respectively.
        d (float): The dilution parameter of the weight matrix. Must be a value
            between 0 and 1, inclusive. A value of 0 means that the weight matrix
            is fully connected, while a value of 1 means that all weights are zero.

    Returns:
        np.ndarray: A 2D numpy array containing the weights of the Hopfield
            network. The element w[i,j] represents the weight between neurons
            i and j in the network.
    """
    # Define weights according to hopfield model for multiple patterns
    n = patterns.shape[0]
    w = np.zeros((n, n))

    # Define a function to compute a single element of the weight matrix
    def compute_weight(i, j):
        return 1/n * np.sum(patterns[i] * patterns[j])

    # Use a ThreadPoolExecutor to compute the weight matrix elements in parallel
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_weight, i, j)
                   for i in range(n) for j in range(n)]
        for i in range(n):
            for j in range(n):
                if random.random() < d:
                    w[i, j] = 0
                else:
                    w[i, j] = futures[i*n+j].result()

    return w


def update_u(u: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Updates the Hopfield network state vector `u` based on the Hopfield weight
    matrix `w`.

    Parameters:
        u (np.ndarray): The Hopfield network state vector of shape (n,), where n
            is the number of neurons in the network
        w (np.ndarray): The Hopfield weight matrix of shape (n, n).

    Returns:
        Tuple[np.ndarray, List[np.ndarray]]: A tuple containing the updated
        Hopfield network state vector of shape (n,) and a list of intermediate
        steps. Each intermediate step is an updated state vector of shape (n,).
    """
    maxiter = 15
    # Initialize the list of intermediate steps with the initial u vector
    intermediate_steps = [u]

    # Compute the first updated u
    u_new = np.sign(np.dot(w, u))
    intermediate_steps.append(u_new)

    # Just to avoid infinite loop
    ii = 0

    # Keep updating u until convergence or max number of iterations is reached
    while not np.array_equal(u_new, u):
        u = u_new
        u_new = np.sign(np.dot(w, u))
        intermediate_steps.append(u_new)
        ii += 1
        if ii > maxiter:
            break

    return u_new, intermediate_steps


def update_u_parallel(u: np.ndarray, w: np.ndarray) -> \
        Tuple[np.ndarray, List[np.ndarray]]:
    """
    Updates the Hopfield network state vector `u` based on the Hopfield weight
    matrix `w`.
    The update is made exploiting parallel computation capabilities.

    Parameters:
        u (np.ndarray): The Hopfield network state vector of shape (n,), where n
            is the number of neurons in the network
        w (np.ndarray): The Hopfield weight matrix of shape (n, n).

    Returns:
        Tuple[np.ndarray, List[np.ndarray]]: A tuple containing the updated
        Hopfield network state vector of shape (n,) and a list of intermediate
        steps. Each intermediate step is an updated state vector of shape (n,).
    """
    maxiter = 15
    # Initialize the list of intermediate steps with the initial u vector
    intermediate_steps = [u]

    # Compute the first updated u
    u_new = np.sign(np.dot(w, u))
    intermediate_steps.append(u_new)

    # Define a function to compute the updated u for a specific intermediate step
    def compute_u(u_prev):
        return np.sign(np.dot(w, u_prev))

    # Just to avoid infinite loop
    ii = 0

    # Keep updating u until convergence or max number of iterations is reached
    with ThreadPoolExecutor() as executor:
        while not np.array_equal(u_new, u):
            u = u_new
            u_new = executor.submit(compute_u, u).result()
            intermediate_steps.append(u_new)
            ii += 1
            if ii > maxiter:
                break

    return u_new, intermediate_steps


def update_u_stochastic(u: np.ndarray, w: np.ndarray, beta: float) -> \
        Tuple[np.ndarray, List[np.ndarray]]:
    """
    Updates the Hopfield network state vector `u` based on the Hopfield weight
    matrix `w`.
    The update is made using a non-deterministic model.

    Parameters:
        u (np.ndarray): The Hopfield network state vector of shape (n,), where n
            is the number of neurons in the network
        w (np.ndarray): The Hopfield weight matrix of shape (n, n).
        beta (float): Inverse to system temperature, controls the shape of the
            updating function tanh()


    Returns:
        Tuple[np.ndarray, List[np.ndarray]]: A tuple containing the updated
        Hopfield network state vector of shape (n,) and a list of intermediate
        steps. Each intermediate step is an updated state vector of shape (n,).
    """
    maxiter = 15
    # Initialize the list of intermediate steps with the initial u vector
    intermediate_steps = [u]

    # Compute the first updated u
    u_new = np.tanh(beta * np.dot(w, u))
    intermediate_steps.append(u_new)

    # Just to avoid infinite loop
    ii = 0

    # Keep updating u until convergence or max number of iterations is reached
    while not np.array_equal(u_new, u):
        u = u_new
        u_new = np.tanh(beta * np.dot(w, u))
        intermediate_steps.append(u_new)
        ii += 1
        if ii > maxiter:
            break

    return u_new, intermediate_steps


def update_u_stochastic_parallel(u: np.ndarray, w: np.ndarray, beta: float) -> \
        Tuple[np.ndarray, List[np.ndarray]]:
    """
    Updates the Hopfield network state vector `u` based on the Hopfield weight
    matrix `w`.
    The update is made using a non-deterministic model and exploiting parallel
    computation capabilities.

    Parameters:
        u (np.ndarray): The Hopfield network state vector of shape (n,), where n
            is the number of neurons in the network
        w (np.ndarray): The Hopfield weight matrix of shape (n, n).
        beta (float): Inverse to system temperature, controls the shape of the
            updating function tanh()


    Returns:
        Tuple[np.ndarray, List[np.ndarray]]: A tuple containing the updated
        Hopfield network state vector of shape (n,) and a list of intermediate
        steps. Each intermediate step is an updated state vector of shape (n,).
    """
    maxiter = 15
    # Initialize the list of intermediate steps with the initial u vector
    intermediate_steps = [u]

    # Compute the first updated u
    u_new = np.tanh(beta * np.dot(w, u))
    intermediate_steps.append(u_new)

    # Define a function to compute the updated u for a specific intermediate step
    def compute_u(u_prev):
        return np.tanh(beta * np.dot(w, u_prev))

    # Just to avoid infinite loop
    ii = 0

    # Keep updating u until convergence or max number of iterations is reached
    with ThreadPoolExecutor() as executor:
        while not np.array_equal(u_new, u):
            u = u_new
            u_new = executor.submit(compute_u, u).result()
            intermediate_steps.append(u_new)
            ii += 1
            if ii > maxiter:
                break

    return u_new, intermediate_steps


def update_u_async(u: np.ndarray, w: np.ndarray) -> \
        Tuple[np.ndarray, List[np.ndarray]]:
    """
    Updates the Hopfield network state vector `u` based on the Hopfield weight
    matrix `w`.
    The update is made in an asynchronous way.

    Parameters:
        u (np.ndarray): The Hopfield network state vector of shape (n,), where n
            is the number of neurons in the network
        w (np.ndarray): The Hopfield weight matrix of shape (n, n).


    Returns:
        Tuple[np.ndarray, List[np.ndarray]]: A tuple containing the updated
        Hopfield network state vector of shape (n,) and a list of intermediate
        steps. Each intermediate step is an updated state vector of shape (n,).
    """
    maxiter = 15
    # Initialize the list of intermediate steps with the initial u vector
    control_steps = [u.copy()]
    intermediate_steps = [u.copy()]

    # Keep updating u until convergence or max number of iterations is reached
    for ii in range(maxiter):
        # Update each element of u asynchronously
        for i in range(len(u)):
            u[i] = np.sign(np.dot(w[i], u))
            # Store intermediate step at the middle of each row
            if not i % (np.sqrt(len(u))/2):
                intermediate_steps.append(u.copy())

        control_steps.append(u.copy())

        # Check for convergence
        if np.array_equal(u, control_steps[-2]):
            break

    return u, intermediate_steps


def update_u_async_stochastic(u: np.ndarray, w: np.ndarray, beta: float) -> \
        Tuple[np.ndarray, List[np.ndarray]]:
    """
    Updates the Hopfield network state vector `u` based on the Hopfield weight
    matrix `w`.
    The update is made in an asynchronous and using a non-deterministic model.

    Parameters:
        u (np.ndarray): The Hopfield network state vector of shape (n,), where n
            is the number of neurons in the network
        w (np.ndarray): The Hopfield weight matrix of shape (n, n).
        beta (float): Inverse to system temperature, controls the shape of the
            updating function tanh()


    Returns:
        Tuple[np.ndarray, List[np.ndarray]]: A tuple containing the updated
        Hopfield network state vector of shape (n,) and a list of intermediate
        steps. Each intermediate step is an updated state vector of shape (n,).
    """
    maxiter = 15
    # Initialize the list of intermediate steps with the initial u vector
    control_steps = [u.copy()]
    intermediate_steps = [u.copy()]

    # Keep updating u until convergence or max number of iterations is reached
    for ii in range(maxiter):
        # Update each element of u asynchronously
        for i in range(len(u)):
            u[i] = np.tanh(beta * np.dot(w[i], u))
            # Store intermediate step at the middle of each row
            if not i % (np.sqrt(len(u))/2):
                intermediate_steps.append(u.copy())

        control_steps.append(u.copy())

        # Check for convergence
        if np.array_equal(u, control_steps[-2]):
            break

    return u, intermediate_steps


def document_test(id_test: str, input_pattern: np.ndarray,
                  process: List[np.ndarray], output: np.ndarray,
                  params: List) -> None:
    """
    Creates a new directory in the tests folder with the given id_test if it
    doesn't exist, and saves plots of the input pattern, output pattern, and
    each element in the process list.
    Additionally, writes a text file to the new directory with information
    about the network size, number of patterns stored, letter to recover,
    noise level beta, and random seed used.

    Args:
    id_test (str): a string that will be used as the name of the new directory.
    input_pattern (np.ndarray): the input pattern used in the test.
    process (List[np.ndarray]): a list of numpy arrays representing the
        intermediate steps in the network's processing.
    output (np.ndarray): the output pattern produced by the network.
    params (List): a list containing various parameters used in the network.

    Returns:
    None
    """
    if not os.path.exists('./tests/' + id_test):
        os.makedirs('./tests/' + id_test)

    # write_pattern_to_file(vector_to_matrix(test), \
    #                       './tests/'+time_string+'input_pattern.txt')
    plot_matrix(vector_to_matrix(input_pattern),
                './tests/' + id_test + '/input_pattern.png')

    # write_pattern_to_file(vector_to_matrix(output), \
    #                       './tests/'+time_string+'output_pattern.txt')
    plot_matrix(vector_to_matrix(output),
                './tests/' + id_test + '/output_pattern.png')

    # Document the test params
    with open('./tests/' + id_test + '/params.txt', 'w') as f:
        # params = [n, letter, beta, seed, async_bool, dilution, len(patterns)]
        f.write('Size of the network: ' + str(params[0]) + ' x ' + str(params[0])
                + ' = ' + str(params[0]*params[0]) + '\n')
        f.write('Number of patterns stored: ' + str(params[6]) + '\n')
        f.write('Letter to recover: ' + params[1] +
                '(if not a vowel, a random input is used)' + '\n')
        f.write('Noise level BETA (=1/T): ' + str(params[2]) + '\n')
        f.write('Random seed used: ' + str(params[3]) + '\n')
        if params[4]:
            f.write('Asynchronous update employed \n')
        if params[5] > 0:
            f.write('Diluted network employed: d=' + str(params[5]) + ' \n')

    # Plot the subsequent patterns during the process
    for ii, step in enumerate(process):
        # write_pattern_to_file(vector_to_matrix(step), \
        #                       './tests/'+time_string+'step_'+str(ii)+'.txt')
        plot_matrix(vector_to_matrix(step),
                    './tests/' + id_test + '/step_' + str(ii) + '.png')
