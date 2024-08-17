import cv2
import numpy as np


def calculate_neighborhood_std(normal_map, kernel_size=3):
    """Calculate the variance of normal vectors in a local neighborhood for each pixel."""
    pad_size = kernel_size // 2
    padded_normals = np.pad(
        normal_map, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="reflect"
    )

    # Extract sliding windows
    neighborhoods = np.lib.stride_tricks.sliding_window_view(
        padded_normals, (kernel_size, kernel_size, 3)
    )
    neighborhoods = neighborhoods.reshape(
        neighborhoods.shape[0], neighborhoods.shape[1], -1
    )

    # Calculate variance over the flattened neighborhood
    variance = np.std(neighborhoods, axis=-1)

    return variance


def detect_flat_areas_by_variance(normal_map, kernel_size=5, percentile=50):
    """Detect flat areas based on variance of normal vectors in a local neighborhood."""
    std = calculate_neighborhood_std(normal_map, kernel_size)

    # Determine adaptive threshold based on image statistics
    variance_threshold = np.percentile(std, percentile)

    # Create a mask where areas with low variance are considered flat
    flat_areas_mask = std < variance_threshold

    # Save mask where flat areas are white (255) and non-flat are black (0)
    flat_areas_mask_img = np.uint8(flat_areas_mask * 255)
    cv2.imwrite("flat_areas_mask_variance.png", flat_areas_mask_img)
    return flat_areas_mask
