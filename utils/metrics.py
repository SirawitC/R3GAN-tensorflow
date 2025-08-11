import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy.linalg import sqrtm

# Standard FID feature extractor (matches pool_3 from original implementation)
_inception_standard = InceptionV3(include_top=False, pooling=None, input_shape=(299, 299, 3))
# Fast mode (directly returns global avg pooled 2048-dim features)
_inception_fast = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
# Inception for IS computation (with classification head)
_inception_classifier = InceptionV3(include_top=True, weights='imagenet')

def _extract_features(images, batch_size=32, method="standard"):
    """
    Extract features from InceptionV3 for FID computation.

    Args:
        images: Tensor or np.array in [0,1] or [0,255]
        batch_size: Batch size for prediction
        method: "standard" or "fast"
    """
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    images = tf.image.resize(images, (299, 299))

    min_val = tf.reduce_min(images)
    max_val = tf.reduce_max(images)

    if min_val < 0.0:  
        # Case: [-1, 1]
        images = (images + 1.0) / 2.0 * 255.0
    elif max_val <= 1.0:  
        # Case: [0, 1]
        images = images * 255.0

    images = preprocess_input(images)

    if method == "standard":
        feats = _inception_standard.predict(images, batch_size=batch_size, verbose=0)
        feats = tf.nn.avg_pool2d(feats, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='valid')
        feats = tf.reshape(feats, (feats.shape[0], -1))
    else:  # fast
        feats = _inception_fast.predict(images, batch_size=batch_size, verbose=0)

    return feats.numpy()

def calculate_fid(real_images, fake_images, batch_size=32, method="standard"):
    """
    Compute Frechet Inception Distance between real and generated images.
    """
    act1 = _extract_features(real_images, batch_size=batch_size, method=method)
    act2 = _extract_features(fake_images, batch_size=batch_size, method=method)

    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

def calculate_inception_score(images, batch_size=32, splits=10):
    """
    Compute Inception Score (mean, std) for generated images in [0,1] or [0,255].
    """
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    images = tf.image.resize(images, (299, 299))

    min_val = tf.reduce_min(images)
    max_val = tf.reduce_max(images)

    if min_val < 0.0:  
        # Case: [-1, 1]
        images = (images + 1.0) / 2.0 * 255.0
    elif max_val <= 1.0:  
        # Case: [0, 1]
        images = images * 255.0

    images = preprocess_input(images)

    preds = _inception_classifier.predict(images, batch_size=batch_size, verbose=0)
    preds = tf.nn.softmax(preds).numpy()

    N = preds.shape[0]
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits):(k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = [np.sum(p * (np.log(p + 1e-16) - np.log(py + 1e-16))) for p in part]
        split_scores.append(np.exp(np.mean(scores)))

    return float(np.mean(split_scores)), float(np.std(split_scores))
