import tensorflow as tf
from trainer import AdversarialTrainer
from models.generator import Generator
from models.discriminator import Discriminator
from tensorflow.keras.datasets import cifar10
from configs.config import *

@tf.function
def train_step(real_images, conditions, noise_dim, gamma=10.0):
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal([batch_size, noise_dim])

    # Train Discriminator
    disc_loss, r1_pen, r2_pen = trainer.train_discriminator_step(noise, real_images, conditions, gamma)

    # Train Generator
    gen_loss = trainer.train_generator_step(noise, real_images, conditions)

    return gen_loss, disc_loss, r1_pen, r2_pen

generator = Generator(NOISE_DIMENSION_G, WIDTH_PER_STAGE_G, CARDINALITY_PER_STAGE_G, BLOCKS_PER_STAGE_G, EXPANSION_FACTOR, CONDITION_DIM, CONDITION_EMBEDDING_DIM_G)
discriminator = Discriminator(WIDTH_PER_STAGE_D, CARDINALITY_PER_STAGE_D, BLOCKS_PER_STAGE_D, EXPANSION_FACTOR, CONDITION_DIM, CONDITION_EMBEDDING_DIM_D)

g_optimizer = tf.keras.optimizers.Adam(LR_G, BETA_1, BETA_2)
d_optimizer = tf.keras.optimizers.Adam(LR_D, BETA_1, BETA_2)

trainer = AdversarialTrainer(generator, discriminator, g_optimizer, d_optimizer)

# Load CIFAR-10
(x_train, y_train), (_, _) = cifar10.load_data()

# Normalize to [-1, 1]
x_train = (x_train.astype("float32") / 127.5) - 1.0

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, CONDITION_DIM)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

train_dataset = train_dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

for epoch in range(NUM_EPOCHS):
    for real_images, conditions in train_dataset:
        gen_loss, disc_loss, r1, r2 = train_step(real_images, conditions, NOISE_DIMENSION_G)
    
    print(f"Epoch {epoch}: G Loss = {gen_loss:.4f}, D Loss = {disc_loss:.4f}, R1 = {r1:.2f}, R2 = {r2:.2f}")
