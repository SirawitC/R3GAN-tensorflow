import tensorflow as tf

class AdversarialTrainer:
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

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
        return tf.stop_gradient(loss)

    def train_discriminator(self, noise, real_images, conditions, gamma=10.0, scale=1.0, preprocessor=lambda x: x):
        real_images = tf.cast(real_images, tf.float32)
        real_images = tf.convert_to_tensor(real_images)

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

            loss = tf.multiply(scale, tf.reduce_mean(tf.add(tf.reshape(tf.multiply(gamma, gp_total), (4, 1)), adversarial_loss)))

        grads = tape.gradient(loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return tf.stop_gradient(loss), tf.stop_gradient(R1_penalty), tf.stop_gradient(R2_penalty)
