import tensorflow as tf
from tensorflow import keras
from component import DiscriminativeBasis, DownsampleLayer, ResidualBlock, MSRInitializer, Convolution

class DiscriminatorStage(keras.Model):
    def __init__(
        self,
        input_channels,
        output_channels,
        cardinality,
        num_blocks,
        expansion_factor,
        kernel_size,
        variance_scaling,
        resampling_filter=None,
        dtype=tf.float32
    ):
        super().__init__()
        if resampling_filter is None:
            transition_layer = DiscriminativeBasis(input_channels, output_channels)
        else:
            transition_layer = DownsampleLayer(input_channels, output_channels, resampling_filter)
        
        self.layers = [
            ResidualBlock(input_channels, cardinality, expansion_factor, kernel_size, variance_scaling) for _ in range(num_blocks)
        ] + [transition_layer]

        self.dtype = dtype

    def call(self, x):
        x = tf.cast(x, self.dtype)
        for layer in self.layers:
            x = layer(x)
        return x

