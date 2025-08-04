import os
import json
import time
import psutil
import tensorflow as tf
import numpy as np

class TrainingLogger:
    """Handles logging operations for GAN training with TensorFlow."""
    
    def __init__(self, run_dir):
        """
        Initialize the training logger.
        
        Args:
            run_dir (str): Directory to save log files
        """
        self.run_dir = run_dir
        self.stats_metrics = dict()
        self.stats_jsonl = None
        self.stats_tfevents = None
        self.current_stats = dict()
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup JSON and TensorBoard logging."""
        print('Initializing logs...')
        
        # Setup JSON logging
        self.stats_jsonl = open(os.path.join(self.run_dir, 'stats.jsonl'), 'wt')
        
        # Setup TensorBoard logging
        try:
            self.stats_tfevents = tf.summary.create_file_writer(self.run_dir)
        except Exception as err:
            print('Skipping tfevents export:', err)
    
    def _format_time(self, seconds):
        """Format time duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m {seconds%60:.0f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes:02d}m"
    
    def _get_gpu_memory_info(self):
        """Get GPU memory information using TensorFlow."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Get memory info from TensorFlow
                memory_info = tf.config.experimental.get_memory_info('GPU:0')
                current_mb = memory_info['current'] / (1024**2)
                peak_mb = memory_info['peak'] / (1024**2)
                return current_mb / 1024, peak_mb / 1024  # Convert to GB
            else:
                return 0.0, 0.0
        except:
            return 0.0, 0.0
    
    def log_progress(self, cur_tick, cur_nimg, total_kimg, tick_start_time, tick_start_nimg, 
                    start_time, maintenance_time, cur_lr, cur_ema_nimg, cur_beta2, cur_gamma, 
                    augment_p=0.0, phase_times=None):
        """
        Log training progress and statistics.
        
        Args:
            cur_tick (int): Current training tick
            cur_nimg (int): Current number of images processed
            total_kimg (int): Total training images in thousands
            tick_start_time (float): Start time of current tick
            tick_start_nimg (int): Number of images at start of tick
            start_time (float): Training start time
            maintenance_time (float): Time spent on maintenance
            cur_lr (float): Current learning rate
            cur_ema_nimg (float): Current EMA parameter
            cur_beta2 (float): Current beta2 parameter
            cur_gamma (float): Current gamma parameter
            augment_p (float): Current augmentation probability
            phase_times (dict): Dictionary of phase names to execution times
        """
        tick_end_time = time.time()
        
        # Calculate timing metrics
        total_time = tick_end_time - start_time
        tick_time = tick_end_time - tick_start_time
        sec_per_kimg = tick_time / (cur_nimg - tick_start_nimg) * 1e3 if cur_nimg > tick_start_nimg else 0
        
        # Get memory info
        cpu_mem_gb = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        gpu_current_gb, gpu_peak_gb = self._get_gpu_memory_info()
        
        # Store stats for later use
        self.current_stats.update({
            'Progress/tick': cur_tick,
            'Progress/kimg': cur_nimg / 1e3,
            'Progress/lr': cur_lr,
            'Progress/ema_mimg': cur_ema_nimg / 1e6,
            'Progress/beta2': cur_beta2,
            'Progress/gamma': cur_gamma,
            'Progress/augment': augment_p,
            'Timing/total_sec': total_time,
            'Timing/sec_per_tick': tick_time,
            'Timing/sec_per_kimg': sec_per_kimg,
            'Timing/maintenance_sec': maintenance_time,
            'Timing/total_hours': total_time / 3600,
            'Timing/total_days': total_time / (24 * 3600),
            'Resources/cpu_mem_gb': cpu_mem_gb,
            'Resources/gpu_mem_gb': gpu_current_gb,
            'Resources/peak_gpu_mem_gb': gpu_peak_gb
        })
        
        # Add phase timing if provided
        if phase_times:
            for phase_name, phase_time in phase_times.items():
                self.current_stats[f'Timing/{phase_name}'] = phase_time
        
        # Build and print status line
        fields = []
        fields += [f"tick {cur_tick:<5d}"]
        fields += [f"kimg {cur_nimg / 1e3:<8.1f}"]
        fields += [f"time {self._format_time(total_time):<12s}"]
        fields += [f"sec/tick {tick_time:<7.1f}"]
        fields += [f"sec/kimg {sec_per_kimg:<7.2f}"]
        fields += [f"maintenance {maintenance_time:<6.1f}"]
        fields += [f"cpumem {cpu_mem_gb:<6.2f}"]
        fields += [f"gpumem {gpu_current_gb:<6.2f}"]
        fields += [f"peak {gpu_peak_gb:<6.2f}"]
        fields += [f"augment {augment_p:.3f}"]
        
        print(' '.join(fields))
    
    def log_losses(self, **losses):
        """
        Log loss values.
        
        Args:
            **losses: Keyword arguments containing loss names and values
        """
        for loss_name, loss_value in losses.items():
            # Convert TensorFlow tensors to Python scalars
            if hasattr(loss_value, 'numpy'):
                loss_value = float(loss_value.numpy())
            elif isinstance(loss_value, np.ndarray):
                loss_value = float(loss_value)
            
            self.current_stats[f'Loss/{loss_name}'] = loss_value
    
    def update_logs(self, cur_nimg, start_time):
        """
        Update JSON and TensorBoard logs.
        
        Args:
            cur_nimg (int): Current number of images processed
            start_time (float): Training start time
        """
        timestamp = time.time()
        
        # Update JSON log
        if self.stats_jsonl is not None:
            fields = dict(self.current_stats, timestamp=timestamp)
            self.stats_jsonl.write(json.dumps(fields) + '\n')
            self.stats_jsonl.flush()
        
        # Update TensorBoard log
        if self.stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            
            with self.stats_tfevents.as_default():
                # Log training statistics
                for name, value in self.current_stats.items():
                    tf.summary.scalar(name, value, step=global_step)
                
                # Log metrics
                for name, value in self.stats_metrics.items():
                    tf.summary.scalar(f'Metrics/{name}', value, step=global_step)
            
            self.stats_tfevents.flush()
    
    def update_metrics(self, result_dict):
        """
        Update metrics dictionary.
        
        Args:
            result_dict: Dictionary containing metric results
        """
        if hasattr(result_dict, 'results'):
            self.stats_metrics.update(result_dict.results)
        else:
            self.stats_metrics.update(result_dict)
    
    def log_images(self, images, step, tag="generated_images", max_outputs=8):
        """
        Log images to TensorBoard.
        
        Args:
            images: Image tensor or numpy array
            step (int): Training step
            tag (str): Tag for the images
            max_outputs (int): Maximum number of images to log
        """
        if self.stats_tfevents is not None:
            # Convert to numpy if needed
            if hasattr(images, 'numpy'):
                images = images.numpy()
            
            # Ensure images are in correct format [batch, height, width, channels]
            if len(images.shape) == 4:
                # Normalize to [0, 1] if needed
                if images.max() > 1.0 or images.min() < 0.0:
                    images = (images + 1.0) / 2.0  # Assuming [-1, 1] range
                
                images = np.clip(images, 0.0, 1.0)
                
                with self.stats_tfevents.as_default():
                    tf.summary.image(tag, images, step=step, max_outputs=max_outputs)
                    self.stats_tfevents.flush()
    
    def close(self):
        """Close log files."""
        if self.stats_jsonl is not None:
            self.stats_jsonl.close()
        if self.stats_tfevents is not None:
            self.stats_tfevents.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

def create_logger(run_dir):
    """
    Create a training logger instance.
    
    Args:
        run_dir (str): Directory to save log files
        
    Returns:
        TrainingLogger: Logger instance
    """
    return TrainingLogger(run_dir)