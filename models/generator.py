import tensorflow as tf
from tensorflow import keras
from component import UpsampleLayer, ResidualBlock, MSRInitializer, Convolution

class GenerativeBasis(keras.layers.Layer):
    def __init__(self, output_channels):
        super().__init__()
        self.basis = self.add_weight(
            shape=(4, 4, output_channels),
            initializer=tf.random_normal_initializer(mean=0., stddev=1.),
            trainable=True,
            name="basis",
            dtype=tf.float32
        )
        self.linear = keras.layers.Dense(
            output_channels, use_bias=False, dtype=tf.float32,
            kernel_initializer=MSRInitializer()
        )

    def call(self, x):
        # x.shape = (4, 128)
        batch_size = tf.shape(x)[0] # 4
        # linear_out.shape = (batch_size, output_channels) = (4, 768)
        linear_out = self.linear(x)
        # (batch_size, output_channels) = (4, 768) -> (batch_size, 1, 1, 768)
        linear_out = tf.reshape(linear_out, (batch_size, 1, 1, -1))
        # basis.shape = (1, 4, 4, 768)
        basis = tf.expand_dims(self.basis, axis=0)

        return basis * linear_out

class GeneratorStage(keras.layers.Layer):
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
            transition_layer = GenerativeBasis(output_channels)
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

    def call(self, x):
        x = tf.cast(x, tf.float32)
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
    ):
        super().__init__()
        self.cond_emb_dim = cond_emb_dim
        self.cond_dim = cond_dim

        variance_scaling = sum(block_per_stage)
        main_layers = [
            GeneratorStage(
                noise_dim + cond_emb_dim, # 64 + 64
                width_per_stage[0], # 768
                cardinality_per_stage[0], # 96
                block_per_stage[0], # 2
                expansion_factor, # 2
                kernel_size, # 3 
                variance_scaling, # 8
                None,
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
                )
            )
        self.main_layers = main_layers
        self.aggregation_layer = Convolution(width_per_stage[-1], 3, kernel_size=1)

        if cond_dim is not None and cond_emb_dim > 0:
            self.embedding_layer = keras.layers.Dense(
                cond_emb_dim, use_bias=False, dtype=tf.float32,
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