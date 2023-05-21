from functions_attractor import *

import datetime
current_time = datetime.datetime.now()
time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")

# Size of the patterns used (n x n) = number of neurons in the network
n = 32
# Letter whose pattern we want to recover
letter = 'A'
# Parameter to control the noise level
beta = 4
# Random seed used
seed = 221
random.seed(seed)
# Whether to use an asynchronous update
async_bool = True
# Whether to use a diluted network
dilution = 0.2
# Group all of them
params = [n, letter, beta, seed, async_bool, dilution]

# Load patterns to store in memory
a_1 = interpolate_pattern('./patterns/n_128/A.txt', n)
e_1 = interpolate_pattern('./patterns/n_128/E.txt', n)
i_1 = interpolate_pattern('./patterns/n_128/I.txt', n)
o_1 = interpolate_pattern('./patterns/n_128/O.txt', n)
u_1 = interpolate_pattern('./patterns/n_128/U.txt', n)

# Reshape those patterns
a = matrix_to_vector(a_1)
e = matrix_to_vector(e_1)
i = matrix_to_vector(i_1)
o = matrix_to_vector(o_1)
u = matrix_to_vector(u_1)

# Include them in the same input to train hopfield weights
# (comment out undesired patterns):
patterns = np.concatenate((
                            a.reshape(-1, 1),
                            e.reshape(-1, 1),
                            i.reshape(-1, 1),
                            o.reshape(-1, 1),
                            u.reshape(-1, 1)
                            ), axis=1)

params.append(patterns.shape[1])

# Train the network weights with the Hopfield Model
w = hopfield_weights(patterns, dilution)


# Test with a random input:
test = random.choices([-1, 1], k=n*n)
if letter == 'A':
    test = np.sign(a + random.choices([-0.95, 1.1], k=n*n)
                     + random.choices([-1.1, 0.95], k=n*n))
elif letter == 'E':
    test = np.sign(e + random.choices([-0.95, 1.1], k=n*n)
                     + random.choices([-1.1, 0.95], k=n*n))
elif letter == 'I':
    test = np.sign(i + random.choices([-0.95, 1.1], k=n*n)
                     + random.choices([-1.1, 0.95], k=n*n))
elif letter == 'O':
    test = np.sign(o + random.choices([-0.95, 1.1], k=n*n)
                     + random.choices([-1.1, 0.95], k=n*n))
elif letter == 'U':
    test = np.sign(u + random.choices([-0.95, 1.1], k=n*n)
                     + random.choices([-1.1, 0.95], k=n*n))

if async_bool:
    output, process = update_u_async_stochastic(test, w, beta)
else:
    output, process = update_u_stochastic_parallel(test, w, beta)

# Document results
document_test(time_string, test, process, output, params)
