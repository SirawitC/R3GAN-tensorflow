import tensorflow as tf
import time
import os
from training.trainer import AdversarialTrainer
from models.generator import Generator
from models.discriminator import Discriminator
from configs.config import *
from tensorflow.keras.datasets import cifar10
from utils.logger import create_logger
from tqdm import tqdm 

@tf.function
def train_step(real_images, conditions, noise_dim, gamma=10.0):
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal([batch_size, noise_dim])

    # Train Discriminator
    d_start = time.time()
    disc_loss, r1_pen, r2_pen = trainer.train_discriminator(noise, real_images, conditions, gamma)
    d_time = time.time() - d_start

    # Train Generator
    g_start = time.time()
    gen_loss = trainer.train_generator(noise, real_images, conditions)
    g_time = time.time() - g_start

    return gen_loss, disc_loss, r1_pen, r2_pen, d_time, g_time

# Create models
generator = Generator(NOISE_DIMENSION_G, WIDTH_PER_STAGE_G, CARDINALITY_PER_STAGE_G, 
                     BLOCKS_PER_STAGE_G, EXPANSION_FACTOR, CONDITION_DIM, CONDITION_EMBEDDING_DIM_G)
discriminator = Discriminator(WIDTH_PER_STAGE_D, CARDINALITY_PER_STAGE_D, BLOCKS_PER_STAGE_D, 
                             EXPANSION_FACTOR, CONDITION_DIM, CONDITION_EMBEDDING_DIM_D)

# Create optimizers
total_steps = int((NUM_EPOCHS * IMAGES_PER_EPOCH) / BATCH_SIZE)

# Cosine decay goes from initial_lr -> alpha * initial_lr
# To match final_lr exactly, set alpha = final_lr / initial_lr
alpha_g = FIN_LR_G / INIT_LR_G
alpha_d = FIN_LR_D / INIT_LR_D

g_lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=INIT_LR_G,
    decay_steps=total_steps,
    alpha=alpha_g
)

d_lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=INIT_LR_D,
    decay_steps=total_steps,
    alpha=alpha_d
)

g_optimizer = tf.keras.optimizers.Adam(
    learning_rate=g_lr_schedule, 
    beta_1=BETA_1, 
    beta_2=BETA_2
)
d_optimizer = tf.keras.optimizers.Adam(
    learning_rate=d_lr_schedule, 
    beta_1=BETA_1, 
    beta_2=BETA_2
)

# Create trainer
trainer = AdversarialTrainer(generator, discriminator, g_optimizer, d_optimizer)

# Setup logging
run_dir = f"runs/gan_training_{int(time.time())}"
os.makedirs(run_dir, exist_ok=True)

# Initialize logger
logger = create_logger(run_dir)

# Training parameters
total_kimg = NUM_EPOCHS * IMAGES_PER_EPOCH // 1000  # Total images in thousands
kimg_per_tick = 50  # Log every 50k images
image_snapshot_ticks = 10  # Save images every 10 ticks
network_snapshot_ticks = 20  # Save model every 20 ticks

# Training state
start_time = time.time()
cur_nimg = 0
cur_tick = 0
tick_start_nimg = 0
tick_start_time = start_time
maintenance_time = 0.0

# Fixed noise for consistent image generation
num_samples = 10
fixed_noise = tf.random.normal([num_samples, NOISE_DIMENSION_G])
# Create fixed conditions (you may need to adjust this based on your condition format)
fixed_conditions = tf.eye(num_samples, CONDITION_DIM)  # Modify as needed

print(f"Starting training for {NUM_EPOCHS} epochs ({total_kimg} kimg total)")
print(f"Logging to: {run_dir}")
print()

# Load CIFAR-10
(x_train, y_train), (_, _) = cifar10.load_data()

# Normalize to [-1, 1]
x_train = (x_train.astype("float32") / 127.5) - 1.0

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, CONDITION_DIM)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

train_dataset = train_dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

try:    
    for epoch in tqdm(range(NUM_EPOCHS)):
        epoch_start_time = time.time()
        
        # Phase timing
        d_times = []
        g_times = []
        
        batch_count = 0
        epoch_losses = {'gen_loss': [], 'disc_loss': [], 'r1_pen': [], 'r2_pen': []}
        
        for real_images, conditions in train_dataset:
            batch_start_time = time.time()
            
            gen_loss, disc_loss, r1, r2, d_time, g_time = train_step(real_images, conditions, NOISE_DIMENSION_G)
            print("gen_loss", gen_loss)
            print("disc_loss", disc_loss)
            print("r1", r1)
            print("r2", r2)
            
            d_times.append(d_time)
            g_times.append(g_time)
            
            # Store losses for epoch averaging
            epoch_losses['gen_loss'].append(float(gen_loss))
            epoch_losses['disc_loss'].append(float(disc_loss))
            epoch_losses['r1_pen'].append(r1[0])
            epoch_losses['r2_pen'].append(r2[0])
            
            # Update image count
            batch_size = tf.shape(real_images)[0]
            cur_nimg += int(batch_size)
            batch_count += 1
            
            # Log every batch (optional, might be too frequent)
            if batch_count % 10 == 0:  # Log every 10 batches
                logger.log_losses(
                    gen_loss=gen_loss,
                    disc_loss=disc_loss,
                    r1_penalty=r1[0],
                    r2_penalty=r2[0]
                )
        
        # End of epoch logging
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        
        # Calculate average losses for the epoch
        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        
        # Calculate average phase times
        avg_d_time = sum(d_times) / len(d_times) if d_times else 0
        avg_g_time = sum(g_times) / len(g_times) if g_times else 0
        
        # Check if we need to log (tick-based logging)
        if cur_nimg >= tick_start_nimg + kimg_per_tick * 1000:
            maintenance_start = time.time()
            
            # Log progress
            logger.log_progress(
                cur_tick=cur_tick,
                cur_nimg=cur_nimg,
                total_kimg=total_kimg,
                tick_start_time=tick_start_time,
                tick_start_nimg=tick_start_nimg,
                start_time=start_time,
                maintenance_time=maintenance_time,
                cur_lr=float(g_optimizer.learning_rate),
                cur_ema_nimg=cur_nimg * 0.5,  # Approximate EMA parameter
                cur_beta2=BETA_2,
                cur_gamma=10.0,  # R1/R2 penalty weight
                augment_p=0.0,  # Set augmentation probability if used
                phase_times={'D': avg_d_time, 'G': avg_g_time}
            )
            
            # Log average losses
            logger.log_losses(**avg_losses)
            
            # Generate and log sample images
            if cur_tick % image_snapshot_ticks == 0:
                try:
                    generated_images = generator(fixed_noise, fixed_conditions, training=False)
                    logger.log_images(generated_images, step=cur_nimg//1000, tag="generated_samples")
                except Exception as e:
                    print(f"Warning: Could not log images: {e}")
            
            # Save model checkpoint
            if cur_tick % network_snapshot_ticks == 0:
                try:
                    checkpoint_dir = os.path.join(run_dir, f"checkpoint_{cur_nimg//1000:06d}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    generator.save_weights(os.path.join(checkpoint_dir, "generator"))
                    discriminator.save_weights(os.path.join(checkpoint_dir, "discriminator"))
                    print(f"Saved checkpoint at {checkpoint_dir}")
                except Exception as e:
                    print(f"Warning: Could not save checkpoint: {e}")
            
            # Update logs
            logger.update_logs(cur_nimg, start_time)
            
            # Update tick state
            cur_tick += 1
            tick_start_nimg = cur_nimg
            tick_start_time = time.time()
            maintenance_time = tick_start_time - maintenance_start
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: "
              f"G Loss = {avg_losses['gen_loss']:.4f}, "
              f"D Loss = {avg_losses['disc_loss']:.4f}, "
              f"R1 = {avg_losses['r1_pen']:.4f}, "
              f"R2 = {avg_losses['r2_pen']:.4f}, "
              f"Time = {epoch_time:.1f}s")

    print("\nTraining completed!")

except KeyboardInterrupt:
    print("\nTraining interrupted by user")

except Exception as e:
    print(f"\nTraining failed with error: {e}")
    raise

finally:
    # Clean up
    logger.close()
    print(f"Logs saved to: {run_dir}")
