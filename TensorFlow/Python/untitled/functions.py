from __future__ import absolute_import, division, print_function

import tensorflow as tf

import numpy as np


def relu(n):  # rectified linear unit
    return max(0, n)


def sigmoid(n):  # sigmoid
    return 1 / (1 - np.exp(n))



def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)


def parse_csv(line):  # parse data
    example_defaults = [[0.], [0]]  # sets field types
    parsed_line = tf.decode_csv(line, example_defaults)
    # First field is a feature
    features = tf.reshape(parsed_line[:-1], shape=(1,))
    # Last field is the label
    label = tf.reshape(parsed_line[-1], shape=())
    return features, label
