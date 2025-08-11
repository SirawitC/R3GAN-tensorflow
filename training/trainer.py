import tensorflow as tf
from configs.config import *
import numpy as np

def ema_decay_from_half_life(step):
        hl_mimg = 0.5 * (1 - tf.cos(np.pi * step / TOTAL_STEPS)) * FIN_EMA_MIMG 

        # convert to images
        hl_images = hl_mimg * 1e6
        images_per_step = BATCH_SIZE

        # decay formula
        decay = tf.pow(0.5, images_per_step / tf.maximum(hl_images, 1.0))
        return decay
class AdversarialTrainer:
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, gamma_schedule):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.gamma_schedule = gamma_schedule
        self.step = tf.Variable(0, dtype=tf.int64, trainable=False)

        # EMA generator weights
        self.generator_ema = tf.keras.models.clone_model(generator)
        self.generator_ema.set_weights(generator.get_weights())
    
    def update_generator_ema(self):
        decay = ema_decay_from_half_life(tf.cast(self.step, tf.float32))
        for w_ema, w in zip(self.generator_ema.weights, self.generator.weights):
            w_ema.assign(w_ema * decay + w * (1.0 - decay))

    @staticmethod
    def zero_centered_gradient_penalty(tape, x, logits):
        pred_sum = tf.reduce_sum(logits)
        grads = tape.gradient(pred_sum, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return tf.reduce_sum(tf.square(grads), axis=[1, 2, 3])

    def train_generator(self, noise, real_images, conditions, scale=1.0, preprocessor=lambda x: x):
        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, conditions, training=True)
            fake_logits = self.discriminator(preprocessor(fake_images), conditions, training=True)
            real_logits = self.discriminator(preprocessor(tf.stop_gradient(real_images)), conditions, training=True)

            relativistic_logits = fake_logits - real_logits
            adversarial_loss = tf.nn.softplus(-relativistic_logits)
            loss = scale * tf.reduce_mean(adversarial_loss)

        grads = tape.gradient(loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        self.update_generator_ema()
        return tf.stop_gradient(loss)

    def train_discriminator(self, noise, real_images, conditions, scale=1.0, preprocessor=lambda x: x):
        real_images = tf.cast(real_images, tf.float32)
        real_images = tf.convert_to_tensor(real_images)

        # Get gamma for this step
        gamma = self.gamma_schedule(self.step)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(real_images)
            fake_images = tf.stop_gradient(self.generator(noise, conditions, training=True))
            
            tape.watch(fake_images)
            real_logits = self.discriminator(preprocessor(real_images), conditions, training=True)
            fake_logits = self.discriminator(preprocessor(fake_images), conditions, training=True)

            relativistic_logits = tf.subtract(real_logits, fake_logits)
            adversarial_loss = tf.nn.softplus(tf.multiply(relativistic_logits, -1))

            R1_penalty = self.zero_centered_gradient_penalty(tape, real_images, real_logits)
            R2_penalty = self.zero_centered_gradient_penalty(tape, fake_images, fake_logits)
            gp_total = tf.multiply(0.5, tf.add(R1_penalty, R2_penalty))

            loss = tf.multiply(scale, tf.reduce_mean(tf.add(tf.reshape(tf.multiply(gamma, gp_total), (BATCH_SIZE, 1)), adversarial_loss)))

        grads = tape.gradient(loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        # Increment step
        self.step.assign_add(1)

        return tf.stop_gradient(loss), tf.stop_gradient(R1_penalty), tf.stop_gradient(R2_penalty)
