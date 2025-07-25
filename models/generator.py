import tensorflow as tf
from tensorflow import keras
from component import GenerativeBasis, UpsampleLayer, ResidualBlock, MSRInitializer, Convolution

class GeneratorStage(keras.Model):
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
            transition_layer = GenerativeBasis(input_channels, output_channels)
        else:
            transition_layer = UpsampleLayer(input_channels, output_channels, resampling_filter)
        
        self.layers = [transition_layer]
        for _ in range(num_blocks):
            self.layers.append(
                ResidualBlock(
                    output_channels, cardinality, expansion_factor,
                    kernel_size, variance_scaling
                )
            )
        self.dtype = dtype

    def call(self, x):
        x = tf.cast(x, self.dtype)
        for layer in self.layers:
            x = layer(x)
        return x
    
class Generator(keras.Model):
    def __init__(
        self,
        noise_dim,
        width_per_stage,
        cardinality_per_stage,
        block_per_stage,
        expansion_factor,
        cond_dim=None,
        cond_emb_dim=0,
        kernel_size=3,
        resampling_filter=[1, 2, 1],
        dtype=tf.float32
    ):
        super().__init__()
        self.cond_emb_dim = cond_emb_dim
        self.cond_dim = cond_dim
        self.dtype = dtype

        variance_scaling = sum(block_per_stage)
        main_layers = [
            GeneratorStage(
                noise_dim + cond_emb_dim,
                width_per_stage[0],
                cardinality_per_stage[0],
                block_per_stage[0],
                expansion_factor,
                kernel_size,
                variance_scaling,
                None,
                dtype=dtype
            )
        ]
        for x in range(len(width_per_stage) - 1):
            main_layers.append(
                GeneratorStage(
                    width_per_stage[x],
                    width_per_stage[x + 1],
                    cardinality_per_stage[x + 1],
                    block_per_stage[x + 1],
                    expansion_factor,
                    kernel_size,
                    variance_scaling,
                    resampling_filter,
                    dtype=dtype
                )
            )
        self.main_layers = main_layers
        self.aggregation_layer = Convolution(width_per_stage[-1], 3, kernel_size=1)

        if cond_dim is not None and cond_emb_dim > 0:
            self.embedding_layer = keras.layers.Dense(
                cond_emb_dim, use_bias=False, dtype=dtype,
                kernel_initializer=MSRInitializer()
            )
        else:
            self.embedding_layer = None

    def call(self, x, y=None):
        if self.embedding_layer is not None and y is not None:
            y_emb = self.embedding_layer(y)
            x = tf.concat([x, y_emb], axis=-1)
        for layer in self.main_layers:
            x = layer(x)
        return