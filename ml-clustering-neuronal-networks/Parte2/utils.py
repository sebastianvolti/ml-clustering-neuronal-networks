import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os



def one_hot_encode(arr, n_labels):

	# Initialize the the encoded array
	one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

	# Fill the appropriate elements with ones
	one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

	# Finally reshape it to get back to the original array
	one_hot = one_hot.reshape((*arr.shape, n_labels))
	return one_hot


def get_batches(arr, batch_size, seq_length):
	batch_size_total = batch_size * seq_length
	# total number of batches we can make, // integer division, round down
	n_batches = len(arr)//batch_size_total

	# Keep only enough characters to make full batches
	arr = arr[:n_batches * batch_size_total]
	# Reshape into batch_size rows, n. of first row is the batch size, the other lenght is inferred
	arr = arr.reshape((batch_size, -1))

	# iterate through the array, one sequence at a time
	for n in range(0, arr.shape[1], seq_length):
	    # The features
	    x = arr[:, n:n+seq_length]
	    # The targets, shifted by one
	    y = np.zeros_like(x)
	    try:
	        y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
	    except IndexError:
	        y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
	    yield x, y 

def get_mode():
	return False