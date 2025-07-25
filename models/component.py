import tensorflow as tf
from tensorflow import keras
import math

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
        if self.activation == 'lrelu':
            self.act_fn = lambda x: tf.nn.leaky_relu(x, alpha=0.2)
            default_gain = math.sqrt(2.0 / (1 + 0.2 ** 2))
        elif self.activation == 'relu':
            self.act_fn = tf.nn.relu
            default_gain = math.sqrt(2.0)
        elif self.activation == 'linear' or self.activation is None:
            self.act_fn = lambda x: x
            default_gain = 1.0
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.gain = gain if gain is not None else default_gain

    def call(self, x):
        # Add bias
        x = tf.nn.bias_add(x, self.bias)
        # Apply activation and gain
        return self.act_fn(x) * self.gain


class Convolution(tf.keras.layers.Layer):
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

###### TODO: finish this class #######
class ResidualBlock(keras.Model):
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
        activation_gain = biased_activation_gain * variance_scaling ** (-1 / (2 * num_linear_layers - 2))

        self.conv1 = Convolution(input_channels, expanded_channels, kernel_size=1, activation_gain=activation_gain)
        self.conv2 = Convolution(expanded_channels, expanded_channels, kernel_size=kernel_size, groups=cardinality, activation_gain=activation_gain)        
        self.conv3 = Convolution(expanded_channels, input_channels, kernel_size=1, activation_gain=0)

