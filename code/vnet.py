import tensorflow as tf


class Block:
    def __init__(self, inputs_shape, n_channels_out, kernel_size, strides, padding):
        self.inputs_shape = inputs_shape
        self.n_channels_out = n_channels_out
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = 1 if padding == "same" else 0

    def get_conv_shape(self):
        """Calculates the output shape following convolution with the
        current block.

        :return: a tuple of the output shape following convolution
        """
        new_shape = (
            int((self.inputs_shape[0] - self.kernel_size + 2 * self.padding) / self.strides[0] + 1),
            int((self.inputs_shape[1] - self.kernel_size + 2 * self.padding) / self.strides[1] + 1),
            int((self.inputs_shape[2] - self.kernel_size + 2 * self.padding) / self.strides[2] + 1),
            self.n_channels_out
        )
        return new_shape

    def get_deconv_shape(self):
        """Calculates the output shape following deconvolution with the
        current block.

        :return: a tuple of the output shape following deconvolution
        """
        new_shape = (
            self.strides[0] * (self.inputs_shape[0] - 1) + self.kernel_size - 2 * self.padding,
            self.strides[1] * (self.inputs_shape[1] - 1) + self.kernel_size - 2 * self.padding,
            self.strides[2] * (self.inputs_shape[2] - 1) + self.kernel_size - 2 * self.padding,
            self.n_channels_out
        )
        return new_shape


class ConvBlock(tf.keras.layers.Layer, Block):
    def __init__(self, n_stages, inputs_shape, n_channels_out, kernel_size=3, strides=(1, 1, 1), padding="same"):
        tf.keras.layers.Layer.__init__(self)
        Block.__init__(self, inputs_shape, n_channels_out, kernel_size, strides, padding)

        ops = []
        for i in range(n_stages):
            if i == 0:
                # Kaiwing He Uniform initialization is default in PyTorch
                ops.append(tf.keras.layers.Conv3D(n_channels_out, kernel_size, strides=strides,
                                                  padding=padding, input_shape=inputs_shape,
                                                  kernel_initializer="he_uniform"))
            else:
                ops.append(tf.keras.layers.Conv3D(n_channels_out, kernel_size, strides=strides, padding="same",
                                                  kernel_initializer="he_uniform"))
            ops.append(tf.keras.layers.BatchNormalization())
            # residual function
            if i != n_stages - 1:
                ops.append(tf.keras.layers.ReLU())

        self.conv = tf.keras.Sequential(ops)
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        # x = (self.conv(x) + x)  # residual function
        x = self.conv(x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(tf.keras.layers.Layer, Block):
    def __init__(self, inputs_shape, n_channels_out, kernel_size=2, strides=(2, 2, 2), padding="valid"):
        tf.keras.layers.Layer.__init__(self)
        Block.__init__(self, inputs_shape, n_channels_out, kernel_size, strides, padding)

        ops = [
            tf.keras.layers.Conv3D(n_channels_out, kernel_size, strides=strides,
                                   padding=padding, input_shape=inputs_shape,
                                   kernel_initializer="he_uniform"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ]

        self.conv = tf.keras.Sequential(ops)

    def call(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(tf.keras.layers.Layer, Block):
    def __init__(self, inputs_shape, n_channels_out, kernel_size=2, strides=(2, 2, 2), padding="valid"):
        tf.keras.layers.Layer.__init__(self)
        Block.__init__(self, inputs_shape, n_channels_out, kernel_size, strides, padding)

        ops = [
            tf.keras.layers.Conv3DTranspose(n_channels_out, kernel_size, strides=strides,
                                            padding=padding, input_shape=inputs_shape,
                                            kernel_initializer="he_uniform"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ]

        self.conv = tf.keras.Sequential(ops)

    def call(self, x):
        x = self.conv(x)
        return x


class OneByOneConvBlock(tf.keras.layers.Layer, Block):
    def __init__(self, inputs_shape, n_channels_out, kernel_size=1, strides=(1, 1, 1), padding="valid"):
        tf.keras.layers.Layer.__init__(self)
        Block.__init__(self, inputs_shape, n_channels_out, kernel_size, strides, padding)

        self.conv = tf.keras.layers.Conv3D(n_channels_out, kernel_size, strides=strides,
                                           padding=padding, input_shape=inputs_shape,
                                           kernel_initializer="he_uniform")

    def call(self, x):
        x = self.conv(x)
        return x


class VNet(tf.keras.Model):
    """Implementation of the VNet backbone for medical image segmentation as
    described by Luo et al. (https://ojs.aaai.org/index.php/AAAI/article/view/17066)
    """

    def __init__(self, inputs_shape, n_classes=1, n_filters=16, has_dropout=True, dropout=0.5):
        super().__init__()
        self.has_dropout = has_dropout

        # left side
        self.block_one = ConvBlock(1, inputs_shape, n_filters)
        self.block_one_dw = DownsamplingConvBlock(self.block_one.get_conv_shape(), n_filters * 2)

        self.block_two = ConvBlock(2, self.block_one_dw.get_conv_shape(), n_filters * 2)
        self.block_two_dw = DownsamplingConvBlock(self.block_two.get_conv_shape(), n_filters * 4)

        self.block_three = ConvBlock(3, self.block_two_dw.get_conv_shape(), n_filters * 4)
        self.block_three_dw = DownsamplingConvBlock(self.block_three.get_conv_shape(), n_filters * 8)

        self.block_four = ConvBlock(3, self.block_three_dw.get_conv_shape(), n_filters * 8)
        self.block_four_dw = DownsamplingConvBlock(self.block_four.get_conv_shape(), n_filters * 16)

        # bottom
        self.block_five = ConvBlock(3, self.block_four_dw.get_conv_shape(), n_filters * 16)
        self.block_five_up = UpsamplingDeconvBlock(self.block_five.get_conv_shape(), n_filters * 8)

        # right side
        self.block_six = ConvBlock(3, self.block_five_up.get_deconv_shape(), n_filters * 8)
        self.block_six_up = UpsamplingDeconvBlock(self.block_six.get_conv_shape(), n_filters * 4)

        self.block_seven = ConvBlock(3, self.block_six_up.get_deconv_shape(), n_filters * 4)
        self.block_seven_up = UpsamplingDeconvBlock(self.block_seven.get_conv_shape(), n_filters * 2)

        self.block_eight = ConvBlock(2, self.block_seven_up.get_deconv_shape(), n_filters * 2)
        self.block_eight_up = UpsamplingDeconvBlock(self.block_eight.get_conv_shape(), n_filters)

        self.block_nine = ConvBlock(1, self.block_eight_up.get_deconv_shape(), n_filters)

        # 1x1x1 convolution
        self.out_conv = OneByOneConvBlock(self.block_nine.get_conv_shape(), n_classes)
        self.out_conv2 = OneByOneConvBlock(self.block_nine.get_conv_shape(), n_classes)

        # regression head
        self.tanh = tf.keras.layers.Activation("tanh", input_shape=self.out_conv.get_conv_shape())

        self.dropout = tf.keras.layers.Dropout(dropout)

    def encoder(self, input_volume):
        x1 = self.block_one(input_volume)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)

        return [x1, x2, x3, x4, x5]

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1

        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)

        # regression task
        out = self.out_conv(x9)
        out_tanh = self.tanh(out)

        # classification task
        out_seg = self.out_conv2(x9)

        return out_tanh, out_seg

    def call(self, input_volume):
        features = self.encoder(input_volume)
        out_tanh, out_seg = self.decoder(features)
        return out_tanh, out_seg


if __name__ == "__main__":
    model = VNet((112, 112, 80, 1))
    model.build(input_shape=(3, 112, 112, 80, 1))
    print(model.summary())
