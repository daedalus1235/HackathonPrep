import numpy as np
import tensorflow as tf


def relu(n):  # rectified linear unit
    return max(0, n)


def sigmoid(n):  # sigmoid
    return 1/(1-np.exp(n))


def l2(v, vp):  # l2 loss
    loss = 0  # type: int
    for n in vp:
        loss += (v[n]-vp[n])**2
    return loss


def mse(v, vp):  # mse loss
    loss = 0
    num = 0
    for n in vp:
        loss += (v[n] - vp[n])**2
        num += 1
    return loss/num


def parse_csv(line):  # parse data
    example_defaults = [[0.], [0.]]  # sets field types
    parsed_line = tf.decode_csv(line, example_defaults)
    # First 4 fields are features, combine into single tensor
    features = tf.reshape(parsed_line[:-1], shape=(1,))
    # Last field is the label
    label = tf.reshape(parsed_line[-1], shape=())
    return features, label

