import tensorflow as tf
from tensorflow import keras
import math
import numpy as np

def create_lowpass_kernel(weights, inplace):
    if inplace:
        kernel = np.array([weights])
    else:
        kernel = np.convolve(weights, [1, 1]).reshape(1, -1)
    kernel = np.matmul(kernel.T, kernel)
    kernel = tf.convert_to_tensor(kernel, dtype=tf.float32)
    return kernel / tf.reduce_sum(kernel)

class InterpolativeUpsampler(keras.layers.Layer):
    def __init__(self, filter_weights):
        super().__init__()
        self.kernel = create_lowpass_kernel(filter_weights, inplace=False)
        self.built_flag = False

    def build(self, input_shape):
        _, h, w, c = input_shape  # batch_size, height, width, channels
        k = self.kernel.shape[0]

        self.conv_transpose = keras.layers.Conv2DTranspose(
            filters=1,  # we'll reshape anyway
            kernel_size=k,
            strides=2,
            padding='same',
            use_bias=False
        )
        self.conv_transpose.build((None, h, w, 1))  # force build
        kernel = 4 * tf.expand_dims(tf.expand_dims(self.kernel, -1), -1)  # (k, k, 1, 1)
        kernel = tf.cast(kernel, tf.float32)
        self.conv_transpose.set_weights([kernel])
        self.built_flag = True

    def call(self, x):
        batch_size, h, w, c = tf.unstack(tf.shape(x))
        x = tf.reshape(x, [batch_size * c, h, w, 1])
        x = self.conv_transpose(x)
        x = tf.reshape(x, [batch_size, h * 2, w * 2, c])
        return x



# class InterpolativeUpsampler(keras.layers.Layer):
#     def __init__(self, filter_weights):
#         super().__init__()
#         self.kernel = create_lowpass_kernel(filter_weights, inplace=False)
#         self.filter_radius = len(filter_weights) // 2

#     def call(self, x):
#         batch_size, height, width, channels = x.shape
#         x_reshaped = tf.reshape(x, [batch_size * channels, height, width, 1])
        
#         # Create kernel with correct shape for conv_transpose2d: (h, w, output_channels, input_channels)
#         kernel = 4 * tf.expand_dims(tf.expand_dims(self.kernel, -1), -1)  # (h, w, 1, 1)
#         kernel = tf.cast(kernel, tf.float32)
        
#         # Apply padding manually to match PyTorch's padding behavior
#         padded_x = tf.pad(x_reshaped, 
#                          [[0, 0], [self.filter_radius, self.filter_radius], 
#                           [self.filter_radius, self.filter_radius], [0, 0]], 
#                          "CONSTANT")

#         # Transposed convolution with stride 2
#         y = tf.nn.conv2d_transpose(
#             padded_x, 
#             kernel,
#             output_shape=[batch_size * channels, height * 2, width * 2, 1],
#             strides=[1, 2, 2, 1],
#             padding='VALID'
#         )
        
#         # Reshape back to original batch structure
#         y = tf.reshape(y, [batch_size, height * 2, width * 2, channels])
#         return y

class InterpolativeDownsampler(keras.layers.Layer):
    def __init__(self, filter_weights):
        super().__init__()
        self.kernel = create_lowpass_kernel(filter_weights, inplace=False)
        self.filter_radius = len(filter_weights) // 2

    def call(self, x):        
        batch_size, height, width, channels = x.shape
        x_reshaped = tf.reshape(x, [batch_size * channels, height, width, 1])
        
        # Create kernel with correct shape: (h, w, input_channels, output_channels)
        kernel = tf.expand_dims(tf.expand_dims(self.kernel, -1), -1)  # (h, w, 1, 1)
        kernel = tf.cast(kernel, x.dtype)
        
        # Apply padding manually
        padded_x = tf.pad(x_reshaped,
                         [[0, 0], [self.filter_radius, self.filter_radius],
                          [self.filter_radius, self.filter_radius], [0, 0]],
                         "CONSTANT")
        
        # Convolution with stride 2
        y = tf.nn.conv2d(
            padded_x,
            kernel,
            strides=[1, 2, 2, 1],
            padding='VALID'
        )
        
        # Reshape back to original batch structure
        y = tf.reshape(y, [batch_size, height // 2, width // 2, channels])
        return y

class InplaceUpsampler(keras.layers.Layer):
    def __init__(self, filter_weights):
        super().__init__()
        self.kernel = create_lowpass_kernel(filter_weights, inplace=True)
        self.filter_radius = len(filter_weights) // 2

    def call(self, x):
        # x: (batch, channels, height, width)
        # First apply pixel shuffle (equivalent to PyTorch's pixel_shuffle)
        # PyTorch pixel_shuffle expects (N, C*r^2, H, W) -> (N, C, H*r, W*r)
        # We need to rearrange channels first
        
        batch_size, channels, height, width = tf.unstack(tf.shape(x))
        
        # Rearrange for pixel shuffle: (batch, channels, height, width) -> (batch, height, width, channels)
        x = tf.transpose(x, [0, 2, 3, 1])
        
        # Apply depth_to_space (equivalent to pixel_shuffle with upscale_factor=2)
        x = tf.nn.depth_to_space(x, 2)  # (batch, height*2, width*2, channels//4)
        
        # Now apply convolution
        new_height, new_width, new_channels = x.shape[1], x.shape[2], tf.shape(x)[3]
        x_reshaped = tf.reshape(x, [batch_size * new_channels, new_height, new_width, 1])
        
        kernel = tf.expand_dims(tf.expand_dims(self.kernel, -1), -1)
        kernel = tf.cast(kernel, x.dtype)
        
        # Apply padding
        padded_x = tf.pad(x_reshaped,
                         [[0, 0], [self.filter_radius, self.filter_radius],
                          [self.filter_radius, self.filter_radius], [0, 0]],
                         "CONSTANT")
        
        y = tf.nn.conv2d(
            padded_x,
            kernel,
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        
        # Reshape back
        y = tf.reshape(y, [batch_size, new_height, new_width, new_channels])
        y = tf.transpose(y, [0, 3, 1, 2])  # Back to (batch, channels, height, width)
        return y

class InplaceDownsampler(keras.layers.Layer):
    def __init__(self, filter_weights):
        super().__init__()
        self.kernel = create_lowpass_kernel(filter_weights, inplace=True)
        self.filter_radius = len(filter_weights) // 2

    def call(self, x):
        # x: (batch, channels, height, width)
        x = tf.transpose(x, [0, 2, 3, 1])  # -> (batch, height, width, channels)
        
        batch_size, height, width, channels = tf.unstack(tf.shape(x))
        x_reshaped = tf.reshape(x, [batch_size * channels, height, width, 1])
        
        kernel = tf.expand_dims(tf.expand_dims(self.kernel, -1), -1)
        kernel = tf.cast(kernel, x.dtype)
        
        # Apply padding
        padded_x = tf.pad(x_reshaped,
                         [[0, 0], [self.filter_radius, self.filter_radius],
                          [self.filter_radius, self.filter_radius], [0, 0]],
                         "CONSTANT")
        
        # Convolution with stride 1
        y = tf.nn.conv2d(
            padded_x,
            kernel,
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        
        # Reshape back to batch structure
        y = tf.reshape(y, [batch_size, height, width, channels])
        
        # Apply pixel unshuffle (equivalent to PyTorch's pixel_unshuffle)
        # space_to_depth with block_size=2
        y = tf.nn.space_to_depth(y, 2)  # (batch, height//2, width//2, channels*4)
        
        y = tf.transpose(y, [0, 3, 1, 2])  # Back to (batch, channels, height, width)
        return y
class MSRInitializer(tf.keras.initializers.Initializer):
    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, shape, dtype=None):
        fan_in = tf.reduce_prod(shape[:-1])
        std = self.gain / tf.math.sqrt(tf.cast(fan_in, tf.float32))
        return tf.random.normal(shape, stddev=std, dtype=dtype)

    def get_config(self):
        return {'gain': self.gain}
    
class BiasedActivation(tf.keras.layers.Layer):
    def __init__(self, input_units, activation='lrelu', gain=None):
        super().__init__()
        self.input_units = input_units

        # Bias parameter
        self.bias = self.add_weight(
            shape=(input_units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )

        # Activation setup
        self.activation = activation.lower()
        self.act_fn = self._get_activation_function(self.activation)
        default_gain = self.get_default_gain(self.activation)
        self.gain = gain if gain is not None else default_gain

    def call(self, x):
        # Add bias
        x = tf.nn.bias_add(x, self.bias)
        # Apply activation and gain
        return self.act_fn(x) * self.gain
    
    @staticmethod
    def get_default_gain(activation):
        activation = activation.lower()
        if activation == 'lrelu':
            return math.sqrt(2.0 / (1 + 0.2 ** 2))
        elif activation == 'relu':
            return math.sqrt(2.0)
        elif activation == 'linear' or activation is None:
            return 1.0
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    @staticmethod
    def _get_activation_function(activation):
        if activation == 'lrelu':
            return lambda x: tf.nn.leaky_relu(x, alpha=0.2)
        elif activation == 'relu':
            return tf.nn.relu
        elif activation == 'linear' or activation is None:
            return lambda x: x
        else:
            raise ValueError(f"Unsupported activation: {activation}")


class Convolution(keras.layers.Layer):
    def __init__(self, input_channels, output_channels, kernel_size, groups=1, activation_gain=1.0):
        super().__init__()
        self.groups = groups
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.activation_gain = activation_gain

        self.conv = tf.keras.layers.Conv2D(
            filters=output_channels,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            groups=groups,
            use_bias=False,
            kernel_initializer=MSRInitializer(gain=activation_gain)
        )

    def call(self, x):
        return self.conv(x)
class ResidualBlock(keras.layers.Layer):
    def __init__(
        self,
        input_channels,
        cardinality,
        expansion_factor,
        kernel_size,
        variance_scaling
        ):
        super().__init__()
        num_linear_layers = 3
        expanded_channels = input_channels * expansion_factor
        activation_gain = BiasedActivation.get_default_gain('lrelu') * variance_scaling ** (-1 / (2 * num_linear_layers - 2))

        self.conv1 = Convolution(input_channels, expanded_channels, kernel_size=1, activation_gain=activation_gain)
        self.conv2 = Convolution(expanded_channels, expanded_channels, kernel_size=kernel_size, groups=cardinality, activation_gain=activation_gain)        
        self.conv3 = Convolution(expanded_channels, input_channels, kernel_size=1, activation_gain=0)

        self.biasact1 = BiasedActivation(expanded_channels)
        self.biasact2 = BiasedActivation(expanded_channels)

    def call(self, x):
        y = self.conv1(x)
        y = self.conv2(self.biasact1(y))
        y = self.conv3(self.biasact2(y))
        
        return x + y
class UpsampleLayer(keras.layers.Layer):
    def __init__(self, input_channels, output_channels, resampling_filter):
        super().__init__()
        self.resampler = InterpolativeUpsampler(resampling_filter)
        self.use_projection = input_channels != output_channels
        if self.use_projection:
            self.linear_layer = Convolution(input_channels, output_channels, kernel_size=1)

    def call(self, x):
        if self.use_projection:
            x = self.linear_layer(x)
        x = self.resampler(x)
        return x

class DownsampleLayer(keras.layers.Layer):
    def __init__(self, input_channels, output_channels, resampling_filter):
        super().__init__()
        self.resampler = InterpolativeDownsampler(resampling_filter)
        self.use_projection = input_channels != output_channels
        if self.use_projection:
            self.linear_layer = Convolution(input_channels, output_channels, kernel_size=1)

    def call(self, x):
        x = self.resampler(x)
        if self.use_projection:
            x = self.linear_layer(x)
        return x
