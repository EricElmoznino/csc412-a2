from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases

install_aliases()

import sys
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam
from autograd.scipy.misc import logsumexp
from autograd.scipy.special import expit as sigmoid

import os
import gzip
import struct
import array

import matplotlib.pyplot as plt
import matplotlib.image
from urllib.request import urlretrieve

from data import load_mnist, plot_images, save_images

# Load MNIST and Set Up Data
n_examples = 100
if n_examples == 100:
    print('DEBUG RUN')
if len(sys.argv) == 1:
    parts = [1, 2, 3, 4]
else:
    parts = sys.argv[1:]
    parts = [int(p) for p in parts]
N_data, train_images, train_labels, test_images, test_labels = load_mnist()
train_images = np.round(train_images[0:n_examples])
train_labels = train_labels[0:n_examples]
test_images = np.round(test_images[0:n_examples])

if 1 in parts:
    print('PART 1')

if 2 in parts:
    print('PART 2')

if 3 in parts:
    print('PART 3')

if 4 in parts:
    print('PART 4')
    # Starter Code for 4d
    # A correct solution here only requires you to correctly write the neglogprob!
    # Because this setup is numerically finicky
    # the default parameterization I've given should give results if neglogprob is correct.
    K = 30
    D = 784

    # Random initialization, with set seed for easier debugging
    # Try changing the weighting of the initial randomization, default 0.01
    init_params = npr.RandomState(0).randn(K, D) * 0.01

    # Implemented batching for you
    batch_size = 10
    num_batches = int(np.ceil(len(train_images) / batch_size))


    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx + 1) * batch_size)


    # This is numerically stable code to for the log of a bernoulli density
    # In particular, notice that we're keeping everything as log, and using logaddexp
    # We never want to take things out of log space for stability
    def bernoulli_log_density(targets, unnormalized_logprobs):
        # unnormalized_logprobs are in R
        # Targets must be 0 or 1
        t2 = targets * 2 - 1
        # Now t2 is -1 or 1, which makes the following form nice
        label_probabilities = -np.logaddexp(0, -unnormalized_logprobs * t2)
        return np.sum(label_probabilities, axis=-1)  # Sum across pixels.


    def batched_loss(params, iter):
        data_idx = batch_indices(iter)
        return neglogprob(params, train_images[data_idx, :])


    def neglogprob(params, data):
        # Implement this as the solution for 4c!
        return


    # Get gradient of objective using autograd.
    objective_grad = grad(batched_loss)


    def print_perf(params, iter, gradient):
        if iter % 30 == 0:
            save_images(sigmoid(params), 'q4plot.png')
            print(batched_loss(params, iter))


    # The optimizers provided by autograd can optimize lists, tuples, or dicts of parameters.
    # You may use these optimizers for Q4, but implement your own gradient descent optimizer for Q3!
    optimized_params = adam(objective_grad, init_params, step_size=0.2, num_iters=10000, callback=print_perf)
