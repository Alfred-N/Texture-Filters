import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from sklearn.cluster import KMeans
import faiss
import platform
import subprocess
import os
import shutil
import tempfile


LUMINANCE_COEFFS = np.array([0.2126, 0.7152, 0.0722])


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(
            f"Function {func.__name__} took {elapsed_time:.6f} seconds ({elapsed_time*1000:.6f} milliseconds)"
        )
        return result

    return wrapper


def kmeans_sklearn(
    img_pixels, k, num_clusters=100, n_init=1, max_iter=1, random_state=42
):
    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    ).fit(img_pixels)
    top_colors = kmeans.cluster_centers_
    return top_colors[:k]


def kmeans_faiss(
    img_pixels, k, num_clusters=100, n_init=1, max_iter=1, gpu=False, random_state=42
):
    d = img_pixels.shape[1]
    kmeans = faiss.Kmeans(
        d,
        num_clusters,
        niter=max_iter,
        nredo=n_init,
        verbose=False,
        gpu=gpu,
        seed=random_state,
    )

    # Remove default behaviour of subsampling the training data
    if gpu:
        kmeans.max_points_per_centroid = 1000000
    else:
        kmeans.cp.max_points_per_centroid = 1000000

    kmeans.train(img_pixels.astype(np.float32))
    centroids = kmeans.centroids

    # Assign each instance to a centroid
    _, I = kmeans.index.search(img_pixels.astype(np.float32), 1)

    # Count the number of instances belonging to each centroid
    counts = np.bincount(I.reshape(-1), minlength=num_clusters)
    centroids_sorted = centroids[np.argsort(-counts)]
    top_colors = centroids_sorted[:k]
    return top_colors


@time_it
def get_topk_colors_kmeans(
    img_np,
    k,
    engine="sklearn",
    gpu=False,
    num_clusters=100,
    n_init=1,
    max_iter=1,
    random_state=42,
):
    img_pixels = img_np.reshape(-1, 3)

    if engine == "sklearn":
        top_colors = kmeans_sklearn(
            img_pixels, k, num_clusters, n_init, max_iter, random_state
        )
    elif engine == "faiss":
        top_colors = kmeans_faiss(
            img_pixels, k, num_clusters, n_init, max_iter, gpu, random_state
        )
    else:
        raise NotImplementedError('Invalid kmeans engine: choose "sklearn"/"faiss"')
    return top_colors


def round_img_to_colors(img_np, colors_arr):
    # Compute the distance between each pixel and each color in the color list
    distances = np.sqrt(
        np.sum(
            (img_np[:, :, np.newaxis, :] - colors_arr[np.newaxis, np.newaxis, :, :])
            ** 2,
            axis=-1,
        )
    )

    # Find the index of the nearest color for each pixel
    nearest_color_index = np.argmin(distances, axis=-1)

    # Replace each pixel with its nearest color
    nearest_color = colors_arr[nearest_color_index]
    return nearest_color.astype(np.uint8), nearest_color_index


def get_luminance(color):
    return np.dot(LUMINANCE_COEFFS, color)


def pairwise_distances(color_set1, color_set2):
    dists = np.zeros((color_set1.shape[0], color_set2.shape[0]))
    for i in range(color_set1.shape[0]):
        for j in range(color_set2.shape[0]):
            dist = np.abs(get_luminance(color_set1[i]) - get_luminance(color_set2[j]))
            dists[i, j] = dist
    return dists


def find_closest_color_indices(color_set1, dist_matrix):
    flat_dist_matrix = dist_matrix.flatten()
    sorted_indices = np.argsort(flat_dist_matrix)
    closest_indices = np.full(color_set1.shape[0], -1)
    available_indices = np.arange(color_set1.shape[0])

    for i in range(len(sorted_indices)):
        set1_index, set2_index = np.unravel_index(sorted_indices[i], dist_matrix.shape)
        if set1_index in available_indices:
            closest_indices[set1_index] = set2_index
            available_indices = np.delete(
                available_indices, np.where(available_indices == set1_index)
            )
            if len(available_indices) == 0:
                break

    return closest_indices


def swap_colors(
    nearest_colors_index_orig,
    topk_colors_orig,
    topk_colors_ref,
    luminance_correction=True,
):
    if luminance_correction:
        dist_mat = pairwise_distances(topk_colors_orig, topk_colors_ref)
        closest_indices = find_closest_color_indices(topk_colors_orig, dist_mat)
        topk_colors_ref_lumin_adjusted = np.take(
            topk_colors_ref, closest_indices, axis=0
        )
        img_swapped = topk_colors_ref_lumin_adjusted[nearest_colors_index_orig]
    else:
        img_swapped = topk_colors_ref[nearest_colors_index_orig]

    img_swapped = img_swapped.astype(np.uint8)
    return img_swapped
