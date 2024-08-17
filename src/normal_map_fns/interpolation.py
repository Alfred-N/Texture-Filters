import cv2
import numpy as np


def normalize_vectors(vectors):
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norm = np.clip(norm, a_min=1e-8, a_max=None)
    return vectors / norm


def apply_linear_smoothing(original_img_np, kernel_size=3):
    img_np = original_img_np.copy()
    pad_size = kernel_size // 2
    padded_img = np.pad(
        img_np, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="reflect"
    )

    neighborhoods = np.lib.stride_tricks.sliding_window_view(
        padded_img, (kernel_size, kernel_size, 3)
    )
    neighborhoods = neighborhoods.reshape(-1, kernel_size * kernel_size, 3)

    # Average the vectors in the neighborhood
    smoothed_vectors = np.mean(neighborhoods, axis=1)

    # Normalize the smoothed vectors to ensure they remain valid normal vectors
    # smoothed_vectors = normalize_vectors(smoothed_vectors)

    smoothed_img = smoothed_vectors.reshape(img_np.shape)

    return smoothed_img
