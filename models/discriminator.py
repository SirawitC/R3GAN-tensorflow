import tensorflow as tf
from tensorflow import keras
from component import DownsampleLayer, ResidualBlock, MSRInitializer, Convolution
import math

class DiscriminativeBasis(keras.layers.Layer):
    def __init__(self, input_channels, output_dim):
        super().__init__()

        self.basis = keras.layers.Conv2D(
            filters=input_channels,
            kernel_size=4,
            strides=1,
            padding='valid',
            groups=input_channels,
            use_bias=False,
            dtype=tf.float32,
            kernel_initializer=MSRInitializer()
        )
        self.linear = keras.layers.Dense(
            output_dim, use_bias=False, dtype=tf.float32,
            kernel_initializer=MSRInitializer()
        )

    def call(self, x):
        out = self.basis(x)  
        out = tf.reshape(out, [tf.shape(x)[0], -1]) 
        return self.linear(out)

class DiscriminatorStage(keras.layers.Layer):
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
    ):
        super().__init__()
        if resampling_filter is None:
            transition_layer = DiscriminativeBasis(input_channels, output_channels)
        else:
            transition_layer = DownsampleLayer(input_channels, output_channels, resampling_filter)
        
        self.layers = [
            ResidualBlock(input_channels, cardinality, expansion_factor, kernel_size, variance_scaling) for _ in range(num_blocks)
        ] + [transition_layer]

    def call(self, x):
        x = tf.cast(x, tf.float32)
        for layer in self.layers:
            x = layer(x)
        return x

class Discriminator(keras.layers.Layer):
    def __init__(
        self,
        width_per_stage,
        cardinality_per_stage,
        block_per_stage,
        expansion_factor,
        cond_dim=None,
        cond_emb_dim=0,
        kernel_size=3,
        resampling_filter=[1, 2, 1],
    ):
        super().__init__()
        variance_scaling = sum(block_per_stage)

        main_layers = [
            DiscriminatorStage(
                width_per_stage[x],
                width_per_stage[x + 1],
                cardinality_per_stage[x],
                block_per_stage[x],
                expansion_factor,
                kernel_size,
                variance_scaling,
                resampling_filter,
                dtype=tf.float32
            )
            for x in range(len(width_per_stage) - 1)
        ]
        main_layers.append(
            DiscriminatorStage(
                width_per_stage[-1],
                1 if cond_dim is None else cond_emb_dim,
                cardinality_per_stage[-1],
                block_per_stage[-1],
                expansion_factor,
                kernel_size,
                variance_scaling,
                None,
                dtype=tf.float32
            )
        )
        self.main_layers = main_layers
        self.extraction_layer = Convolution(3, width_per_stage[0], kernel_size=1)

        if cond_dim is not None and cond_emb_dim > 0:
            self.embedding_layer = keras.layers.Dense(
                cond_emb_dim, use_bias=False, dtype=tf.float32,
                kernel_initializer=MSRInitializer(gain=1 / math.sqrt(cond_emb_dim))
            )
        else:
            self.embedding_layer = None

    def call(self, x, y=None):
        x = self.extraction_layer(tf.cast(x, tf.float32))
        for layer in self.main_layers:
            x = layer(x)
        if self.embedding_layer is not None and y is not None:
            y_emb = self.embedding_layer(y)
            # Expand y_emb to match x's spatial dims for broadcasting
            y_emb = tf.expand_dims(tf.expand_dims(y_emb, axis=1), axis=1)
            x = tf.reduce_sum(x * y_emb, axis=[1, 2], keepdims=True)
        return tf.squeeze(x, axis=[1, 2, -1])