import os
import tensorflow as tf

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


class ConvBlock(tf.Module):
    def __init__(self, n_stages, n_channels_out, input_shape, kernel_size=3, strides=1):
        super(ConvBlock, self).__init__()

        layers = []
        for i in range(n_stages):
            if i == 0:
                layers.append(tf.keras.layers.Conv3D(n_channels_out, kernel_size, strides=strides,
                                                     padding="same", input_shape=input_shape))
            else:
                layers.append(tf.keras.layers.Conv3D(n_channels_out, kernel_size, strides=strides, padding="same"))
            layers.append(tf.keras.layers.BatchNormalization())
            # residual function
            if i != n_stages - 1:
                layers.append(tf.keras.layers.PReLU())

        self.conv = tf.keras.Sequential(layers)
        self.relu = tf.keras.layers.PReLU()

    def __call__(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(tf.Module):
    def __init__(self):
        super(DownsamplingConvBlock, self).__init__()

    def __call__(self, x):
        pass
