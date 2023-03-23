import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from sklearn.cluster import KMeans
import faiss

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
    # top_colors = centroids[:k]
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


def downsample_image(img_np, downsample_ratio):
    # Calculate the new dimensions of the image
    new_width = int(img_np.shape[1] * downsample_ratio)
    new_height = int(img_np.shape[0] * downsample_ratio)
    return cv2.resize(img_np, (new_width, new_height))


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
    # Flatten the distance matrix
    flat_dist_matrix = dist_matrix.flatten()

    # Sort and argsort the flattened distance matrix
    sorted_indices = np.argsort(flat_dist_matrix)
    # sorted_distances = flat_dist_matrix[sorted_indices]

    # Create arrays to store the closest matching indices and available indices
    closest_indices = np.full(color_set1.shape[0], -1)
    available_indices = np.arange(color_set1.shape[0])

    # Loop through the distances from smallest to largest
    for i in range(len(sorted_indices)):
        # Get the indices of the colors in color_set1 and color_set2
        set1_index, set2_index = np.unravel_index(sorted_indices[i], dist_matrix.shape)

        # Check if the closest color in color_set1 is available
        if set1_index in available_indices:
            # Assign the closest index to the color in color_set2
            closest_indices[set1_index] = set2_index

            # Remove the closest index from the available indices
            available_indices = np.delete(
                available_indices, np.where(available_indices == set1_index)
            )

            # Stop looping if all indices have been used
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

        # Replace each pixel with its nearest color
        img_swapped = topk_colors_ref_lumin_adjusted[nearest_colors_index_orig]
    else:
        img_swapped = topk_colors_ref[nearest_colors_index_orig]

    img_swapped = img_swapped.astype(np.uint8)
    return img_swapped


def plot_filtered_and_original(
    orig_np, filtered_np, color_palette, k, ref_np=None, savepath="comparison.jpg"
):
    fig = plt.figure(figsize=(8, 6))
    if ref_np is not None:
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[5, 1])
    else:
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[5, 1])

    # Plot the original image on the left
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_np[:, :, ::-1].astype(np.uint8))
    ax1.set_title("Original Image")

    # Plot the filtered image on the right
    ax2 = fig.add_subplot(gs[0, 1] if ref_np is not None else gs[0, 1])
    ax2.set_title("Rounded Image")
    ax2.imshow(filtered_np[:, :, ::-1].astype(np.uint8))

    # If a reference image is provided, plot it in the center
    if ref_np is not None:
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(ref_np[:, :, ::-1].astype(np.uint8))
        ax3.set_title("Reference Image")

    # Plot the color palette below the images
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_title(f"Top {k} Colors")
    ax4.imshow(color_palette[:, :, ::-1].astype(np.uint8), aspect=1 / 5)
    ax4.set_axis_off()
    plt.tight_layout()
    plt.savefig(savepath)


def main(
    input_path,
    output_path,
    ref_path=None,
    n_colors=4,
    num_clusters=100,
    downsample_ratio=1,
    random_state=42,
    plot_comparison=False,
    luminance_correction=True,
    engine="sklearn",
    gpu=False,
    n_init=1,
):
    # Sanity checks
    assert not (gpu and engine == "sklearn"), "Only faiss is compatible with gpu"

    # Read the image using OpenCV
    img = cv2.imread(input_path)
    img_np = np.array(img)
    if ref_path is not None:
        ref_img = cv2.imread(ref_path)
        ref_np = np.array(ref_img)
    else:
        ref_np = None

    print("K-means clustering input image ...")
    top_k_colors = get_topk_colors_kmeans(
        img_np,
        n_colors,
        num_clusters=num_clusters,
        random_state=random_state,
        engine=engine,
        gpu=gpu,
        n_init=n_init,
    )
    result_np, nearest_colors_inds_orig = round_img_to_colors(img, top_k_colors)
    if ref_path is not None:
        print("K-means clustering reference image ...")
        top_k_colors_ref = get_topk_colors_kmeans(
            ref_np,
            n_colors,
            num_clusters=num_clusters,
            random_state=random_state,
            engine=engine,
            gpu=gpu,
            n_init=n_init,
        )
        result_np = swap_colors(
            nearest_colors_inds_orig,
            top_k_colors,
            top_k_colors_ref,
            luminance_correction=luminance_correction,
        )

    result_np = downsample_image(result_np, downsample_ratio)

    cv2.imwrite(output_path, result_np)
    print(f"Result saved to {output_path}")

    if plot_comparison:
        savepath_comparison = "comparison.jpg"
        color_palette = np.zeros((1, n_colors, 3))
        if ref_path is not None:
            color_palette[0, :, :] = top_k_colors_ref
        else:
            color_palette[0, :, :] = top_k_colors
        plot_filtered_and_original(
            img_np,
            result_np,
            color_palette,
            n_colors,
            ref_np,
            savepath=savepath_comparison,
        )
        print(f"Comparison plot saved to {savepath_comparison}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform color reduction using k-means clustering."
    )

    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        required=True,
        help="Path to the input image file.",
    )

    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output image file.",
    )

    parser.add_argument(
        "--ref_path", type=str, default=None, help="Path to the reference image file."
    )

    parser.add_argument(
        "--n_colors",
        type=int,
        default=4,
        help="Number of colors to use in the output image.",
    )

    parser.add_argument(
        "--num_clusters",
        type=int,
        default=100,
        help="Number of clusters to use for k-means clustering.",
    )

    parser.add_argument(
        "--n_random",
        type=int,
        default=1,
        help="Number of different random seeds to start from for k-means clustering (increases time linearly).",
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=1,
        help="Number of iterations to use for k-means clustering.",
    )

    parser.add_argument(
        "--downsample_ratio",
        type=int,
        default=1,
        help="Downsampling ratio for the output image.",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for k-means clustering.",
    )

    parser.add_argument(
        "--plot_comparison",
        action="store_true",
        help="Whether to plot a comparison of the original and filtered images.",
    )

    parser.add_argument(
        "--no_luminance_correction",
        dest="luminance_correction",
        action="store_false",
        help="Disable luminance correction when swapping colors.",
    )

    parser.add_argument(
        "--engine",
        choices=["faiss", "sklearn"],
        default="sklearn",
        help="The clustering engine to use",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help='Use gpu for kmeans clustering. Only compatible with engine="faiss"',
    )

    args = parser.parse_args()

    main(
        args.input_path,
        args.output_path,
        ref_path=args.ref_path,
        n_colors=args.n_colors,
        num_clusters=args.num_clusters,
        downsample_ratio=args.downsample_ratio,
        random_state=args.random_state,
        plot_comparison=args.plot_comparison,
        luminance_correction=args.luminance_correction,
        engine=args.engine,
        gpu=args.gpu,
        n_init=args.n_random,
    )
