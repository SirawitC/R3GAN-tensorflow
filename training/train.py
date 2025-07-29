import tensorflow as tf
from trainer import AdversarialTrainer
from models.generator import Generator
from models.discriminator import Discriminator
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

######## TODO: load the appropriate train_dataset #########
for epoch in range(NUM_EPOCHS):
    for real_images, conditions in train_dataset:
        gen_loss, disc_loss, r1, r2 = train_step(real_images, conditions, NOISE_DIMENSION_G)
    
    print(f"Epoch {epoch}: G Loss = {gen_loss:.4f}, D Loss = {disc_loss:.4f}, R1 = {r1:.2f}, R2 = {r2:.2f}")
