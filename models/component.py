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
