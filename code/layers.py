import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import HeNormal, Constant

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


def convolution(x, filter, padding="SAME", strides=None):
    w = tf.get_variable(name="weights", initializer=HeNormal)
    b = tf.get_variable(name="biases", initializer=Constant)
    return tf.nn.convolution(x, w, padding, strides) + b


def deconvolution(x, filter, output_shape, strides=None):
    pass
