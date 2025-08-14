import tensorflow as tf
import time
import numpy as np
import os
from training.trainer import AdversarialTrainer
from models.generator import Generator
from models.discriminator import Discriminator
from configs.config import *
from tensorflow.keras.datasets import cifar10
from utils.logger import create_logger
# from utils.metrics import calculate_fid, calculate_inception_score 

import warnings
warnings.filterwarnings('ignore') 

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU(s) detected:")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU detected by TensorFlow.")

@tf.function
def train_step(trainer, real_images, conditions, noise_dim):
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal([batch_size, noise_dim])
    
    # Train Discriminator
    d_start = time.time()
    disc_loss, r1_pen, r2_pen = trainer.train_discriminator(noise, real_images, conditions)
    d_time = time.time() - d_start

    # Train Generator
    g_start = time.time()
    gen_loss = trainer.train_generator(noise, real_images, conditions)
    g_time = time.time() - g_start

    return gen_loss, disc_loss, r1_pen, r2_pen, d_time, g_time

def increasing_cosine_scheduler(step):
    cosine = 0.5 * (1 - tf.cos(np.pi * step / TOTAL_STEPS))
    return INIT_BETA_2 + (FIN_BETA_2 - INIT_BETA_2) * cosine

# Create models
generator = Generator(NOISE_DIMENSION_G, WIDTH_PER_STAGE_G, CARDINALITY_PER_STAGE_G, 
                     BLOCKS_PER_STAGE_G, EXPANSION_FACTOR, CONDITION_DIM, CONDITION_EMBEDDING_DIM_G)
discriminator = Discriminator(WIDTH_PER_STAGE_D, CARDINALITY_PER_STAGE_D, BLOCKS_PER_STAGE_D, 
                             EXPANSION_FACTOR, CONDITION_DIM, CONDITION_EMBEDDING_DIM_D)

dummy_noise = tf.zeros([1, NOISE_DIMENSION_G])
dummy_condition = tf.zeros([1, CONDITION_DIM])
temp = generator(dummy_noise, dummy_condition, training=False)

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
    beta_2=0.9,
    epsilon=1e-8
)

d_optimizer = tf.keras.optimizers.Adam(
    learning_rate=d_lr_schedule, 
    beta_1=BETA_1, 
    beta_2=0.9,
    epsilon=1e-8
)

# Cosine decay for gamma (R1/R2 penalty strength)
gamma_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=INIT_GAMMA,          
    decay_steps=total_steps,
    alpha=FIN_GAMMA / INIT_GAMMA                       
)

# Create trainer
trainer = AdversarialTrainer(generator, discriminator, g_optimizer, d_optimizer, gamma_schedule)

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
num_samples = 16
fixed_noise = tf.random.normal([num_samples, NOISE_DIMENSION_G])
# Create fixed conditions (you may need to adjust this based on your condition format)
fixed_conditions = tf.eye(num_samples, CONDITION_DIM)  # Modify as needed

# Parameters to control fid calculation
num_fake_sample = 10000

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

train_dataset = train_dataset.shuffle(50000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

try:    
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # Phase timing
        d_times = []
        g_times = []
        
        batch_count = 0
        epoch_losses = {'gen_loss': [], 'disc_loss': [], 'r1_pen': [], 'r2_pen': []}

        for real_images, conditions in train_dataset:
            batch_start_time = time.time()
            
            gen_loss, disc_loss, r1, r2, d_time, g_time = train_step(trainer, real_images, conditions, NOISE_DIMENSION_G)
            
            trainer.g_optimizer.beta_2 = tf.cast(increasing_cosine_scheduler(int(trainer.step)), tf.float32)
            trainer.d_optimizer.beta_2 = tf.cast(increasing_cosine_scheduler(int(trainer.step)), tf.float32)
            
            d_times.append(d_time)
            g_times.append(g_time)
            
            # Store losses for epoch averaging
            epoch_losses['gen_loss'].append(float(gen_loss))
            epoch_losses['disc_loss'].append(float(disc_loss))
            epoch_losses['r1_pen'].append(float(r1[0]))
            epoch_losses['r2_pen'].append(float(r2[0]))
            
            # Update image count
            batch_size = tf.shape(real_images)[0]
            cur_nimg += int(batch_size)
            batch_count += 1
            
            # Log every batch (optional, might be too frequent)
            if batch_count % 10 == 0:  # Log every 10 batches
                logger.log_losses(
                    gen_loss=float(gen_loss),
                    disc_loss=float(disc_loss),
                    r1_penalty=float(r1[0]),
                    r2_penalty=float(r2[0])
                )
            if batch_count % 500 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} (batch {batch_count})| gen_loss: {float(gen_loss)}, disc_loss: {float(disc_loss)}, r1: {float(r1[0])}, r2: {float(r2[0])}")            
        
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
                cur_beta2=float(trainer.g_optimizer.beta_2),
                cur_gamma=float(trainer.gamma_schedule(trainer.step)),  # R1/R2 penalty weight
                augment_p=0.0,  # Set augmentation probability if used
                phase_times={'D': float(avg_d_time), 'G': float(avg_g_time)}
            )
            
            # Log average losses
            logger.log_losses(**avg_losses)

            # Generate 10,000 fake images and compute FID and IS score (This can cause OOM)
            # if epoch % 10 == 0:
            #     fid_noise = tf.random.normal([num_fake_sample, NOISE_DIMENSION_G])
            #     fid_cond = tf.tile(tf.eye(10, 10), multiples=[num_fake_sample//10, 1])
            #     fid_fake_images = trainer.generator_ema(fid_noise, fid_cond, training=False)
            #     fid_real_images = x_train
                
            #     is_mean, is_std = calculate_inception_score(fid_fake_images)
            #     fid = calculate_fid(fid_real_images, fid_fake_images)
            #     logger.log_metrics({"FID": fid, "IS_mean": is_mean, "IS_std": is_std})
            
            # Generate and log sample images
            if epoch % 10 == 0:
                try:
                    generated_images = trainer.generator_ema(fixed_noise, fixed_conditions, training=False)
                    logger.log_images(generated_images, step=cur_nimg//1000, tag="generated_samples")
                except Exception as e:
                    print(f"Warning: Could not log images: {e}")
            
            # Save model checkpoint
            if cur_tick % network_snapshot_ticks == 0:
                try:
                    checkpoint_dir = os.path.join(run_dir, f"checkpoint_{cur_nimg//1000:06d}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    trainer.generator_ema.save_weights(os.path.join(checkpoint_dir, "generator_ema"))
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
        print("="*20)

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