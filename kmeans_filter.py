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
from tqdm import tqdm

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


def plot_filtered_and_original(
    orig_np, filtered_np, color_palette, k, ref_np=None, savepath="comparison.jpg"
):
    fig = plt.figure(figsize=(8, 6))
    if ref_np is not None:
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[5, 1])
    else:
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[5, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_np[:, :, ::-1].astype(np.uint8))
    ax1.set_title("Original Image")

    ax2 = fig.add_subplot(gs[0, 1] if ref_np is not None else gs[0, 1])
    ax2.set_title("Rounded Image")
    ax2.imshow(filtered_np[:, :, ::-1].astype(np.uint8))

    if ref_np is not None:
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(ref_np[:, :, ::-1].astype(np.uint8))
        ax3.set_title("Reference Image")

    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_title(f"Top {k} Colors")
    ax4.imshow(color_palette[:, :, ::-1].astype(np.uint8), aspect=1 / 5)
    ax4.set_axis_off()
    plt.tight_layout()
    plt.savefig(savepath)


def convert_exr_to_png(exr_path):
    if platform.system() == "Darwin":
        png_path = exr_path.replace(".exr", ".png")
        command = ["sips", "-s", "format", "png", exr_path, "--out", png_path]
        try:
            subprocess.run(command, check=True)
            print(f"Converted {exr_path} to {png_path}")
            return png_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to convert EXR to PNG: {e}")
    else:
        raise NotImplementedError("EXR format is only supported on macOS using sips.")


def convert_png_to_exr(png_path):
    if platform.system() == "Darwin":
        exr_path = png_path.replace(".png", ".exr")
        command = ["sips", "-s", "format", "exr", png_path, "--out", exr_path]
        try:
            subprocess.run(command, check=True)
            print(f"Converted {png_path} to {exr_path}")
            return exr_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to convert PNG to EXR: {e}")
    else:
        raise NotImplementedError(
            "PNG to EXR conversion is only supported on macOS using sips."
        )


def load_image(input_path):
    if input_path.endswith(".exr"):
        input_path = convert_exr_to_png(input_path)
    img_np = cv2.imread(input_path)
    if img_np is None:
        raise ValueError(f"Failed to load image from {input_path}")
    return img_np, input_path


def apply_smoothing(img_np, smoothing_type="gaussian", smoothing_strength=5):
    if smoothing_type == "gaussian":
        img_np = cv2.GaussianBlur(img_np, (smoothing_strength, smoothing_strength), 0)
    elif smoothing_type == "bilateral":
        img_np = cv2.bilateralFilter(
            img_np, d=9, sigmaColor=smoothing_strength, sigmaSpace=smoothing_strength
        )
    elif smoothing_type == "median":
        img_np = cv2.medianBlur(img_np, smoothing_strength)
    else:
        raise ValueError(f"Unknown smoothing type: {smoothing_type}")
    return img_np


def apply_bilateral_filter(img_np, d=15, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img_np, d, sigma_color, sigma_space)


def apply_median_filter(img_np, ksize=5):
    return cv2.medianBlur(img_np, ksize)


def apply_gaussian_blur_and_edges(
    img_np, blur_ksize=7, canny_threshold1=50, canny_threshold2=150
):
    blurred_img = cv2.GaussianBlur(img_np, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blurred_img, canny_threshold1, canny_threshold2)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(blurred_img, edges_colored)


def apply_cartoon_effect(img_np):
    img_color = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    img_edges = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2
    )
    img_edges_colored = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR)
    img_cartoon = cv2.bitwise_and(img_color, img_edges_colored)

    return img_cartoon


def apply_cartoon_effect_v2(img_np):
    img_color = cv2.bilateralFilter(img_np, d=15, sigmaColor=100, sigmaSpace=100)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    img_edges = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2
    )
    edges_blur = cv2.GaussianBlur(img_edges, (3, 3), 0)
    edges_colored = cv2.cvtColor(edges_blur, cv2.COLOR_GRAY2BGR)
    img_color_smoothed = cv2.bilateralFilter(
        img_color, d=25, sigmaColor=150, sigmaSpace=150
    )
    img_cartoon = cv2.addWeighted(img_color_smoothed, 0.9, edges_colored, 0.1, 0)

    return img_cartoon


def taylor_exp(x):
    """Taylor series approximation of exp(x) up to the 3rd degree."""
    return 1 + x + (x**2)/2 + (x**3)/6

def gaussian(x, sigma):
    """Gaussian function with Taylor series approximation for the exponential."""
    taylor_x = -(x ** 2) / (2 * sigma ** 2)
    return taylor_exp(taylor_x) / (np.sqrt(2 * np.pi) * sigma)

def compute_structure_tensor(image):
    # Convert to grayscale if the image is multi-channel (RGB)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    Sx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Sy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    Sxx = Sx * Sx
    Syy = Sy * Sy
    Sxy = Sx * Sy
    return Sxx, Syy, Sxy

def anisotropic_kuwahara_filter(image, kernel_size=5, alpha=1.0, blur_radius=2, zero_crossing=0.58):
    height, width = image.shape[:2]
    
    # Compute structure tensor
    Sxx, Syy, Sxy = compute_structure_tensor(image)
    
    # Gaussian blur the structure tensor
    Sxx_blur = cv2.GaussianBlur(Sxx, (blur_radius*2+1, blur_radius*2+1), 0)
    Syy_blur = cv2.GaussianBlur(Syy, (blur_radius*2+1, blur_radius*2+1), 0)
    Sxy_blur = cv2.GaussianBlur(Sxy, (blur_radius*2+1, blur_radius*2+1), 0)
    
    # Compute anisotropy
    lambda1 = 0.5 * (Sxx_blur + Syy_blur + np.sqrt((Sxx_blur - Syy_blur) ** 2 + 4 * Sxy_blur ** 2))
    lambda2 = 0.5 * (Sxx_blur + Syy_blur - np.sqrt((Sxx_blur - Syy_blur) ** 2 + 4 * Sxy_blur ** 2))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        A = np.divide(lambda1 - lambda2, lambda1 + lambda2, where=(lambda1 + lambda2) != 0)
        A = np.nan_to_num(A, nan=0.0)

    phi = 0.5 * np.arctan2(2 * Sxy_blur, Sxx_blur - Syy_blur)
    
    result = np.zeros_like(image, dtype=np.float64)
    
    max_radius = kernel_size // 2
    y_indices, x_indices = np.meshgrid(np.arange(-max_radius, max_radius + 1), np.arange(-max_radius, max_radius + 1), indexing='ij')
    v = np.stack([y_indices.ravel(), x_indices.ravel()], axis=-1)

    for y in tqdm(range(height), desc="Applying Anisotropic Kuwahara Filter"):
        for x in range(width):
            phi_value = phi[y, x]
            cos_phi = np.cos(phi_value)
            sin_phi = np.sin(phi_value)
            R = np.array([[cos_phi, -sin_phi], [sin_phi, cos_phi]])
            
            A_value = A[y, x]
            a = max_radius * np.clip((alpha + A_value) / alpha, 0.1, 2.0)
            b = max_radius * np.clip(alpha / (alpha + A_value), 0.1, 2.0)
            
            S = np.array([[0.5 / a, 0], [0, 0.5 / b]])
            SR = S @ R

            v_transformed = np.dot(v, SR.T)
            inside_mask = np.linalg.norm(v_transformed, axis=1) <= 0.5
            
            y_offsets = np.clip(y + y_indices.ravel()[inside_mask], 0, height - 1)
            x_offsets = np.clip(x + x_indices.ravel()[inside_mask], 0, width - 1)

            region_values = image[y_offsets, x_offsets]
            mean_value = np.mean(region_values, axis=0)
            result[y, x] = mean_value
    
    return np.clip(result, 0, 255).astype(np.uint8)


def main(
    input_path,
    output_path=None,
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
    smoothing_type=None,
    smoothing_strength=5,
    post_process=None,
    anisotropic_kuwahara=False,  # New argument to enable anisotropic Kuwahara filter
    kuwahara_kernel_size=5,  # New argument for kernel size of anisotropic Kuwahara
    kuwahara_alpha=1.0,  # New argument for alpha parameter of anisotropic Kuwahara
    kuwahara_blur_radius=2,  # New argument for blur radius of anisotropic Kuwahara
    kuwahara_zero_crossing=0.58,  # New argument for zero crossing of anisotropic Kuwahara
):
    # Sanity checks
    assert not (gpu and engine == "sklearn"), "Only faiss is compatible with gpu"

    # Load the input image, converting if necessary
    img_np, converted_input_path = load_image(input_path)
    input_is_exr = input_path.endswith(".exr")

    if ref_path is not None:
        ref_np, _ = load_image(ref_path)
    else:
        ref_np = None

    if smoothing_type:
        img_np = apply_smoothing(img_np, smoothing_type, smoothing_strength)

    # Apply Anisotropic Kuwahara Filter if enabled
    if anisotropic_kuwahara:
        print("Applying Anisotropic Kuwahara Filter ...")
        img_np = anisotropic_kuwahara_filter(
            img_np,
            kernel_size=kuwahara_kernel_size,
            alpha=kuwahara_alpha,
            blur_radius=kuwahara_blur_radius,
            zero_crossing=kuwahara_zero_crossing,
        )

    # If output path is not specified, generate one
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_extension = ".exr" if input_is_exr else ".png"
        output_path = os.path.join(
            os.path.dirname(input_path), f"{base_name}_stylized{output_extension}"
        )
    else:
        if input_is_exr and not output_path.endswith(".exr"):
            output_path = output_path.replace(".jpg", ".exr").replace(".png", ".exr")

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
    result_np, nearest_colors_inds_orig = round_img_to_colors(img_np, top_k_colors)
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

    if post_process:
        if post_process == "bilateral":
            result_np = apply_bilateral_filter(result_np)
        elif post_process == "median":
            result_np = apply_median_filter(result_np)
        elif post_process == "gaussian_edges":
            result_np = apply_gaussian_blur_and_edges(result_np)
        elif post_process == "cartoon":
            result_np = apply_cartoon_effect(result_np)
        elif post_process == "cartoon2":
            result_np = apply_cartoon_effect_v2(result_np)
        else:
            raise ValueError(f"Unknown post-processing option: {post_process}")

    output_png_path = (
        output_path
        if output_path.endswith(".png")
        else output_path.replace(".exr", ".png")
    )
    cv2.imwrite(output_png_path, result_np)
    print(f"Result saved to {output_png_path}")

    if input_is_exr:
        output_path = convert_png_to_exr(output_png_path)

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
        help="Path to save the output image file. If not specified, the output will be saved as <input_name>_stylized.[png|exr]",
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

    parser.add_argument(
        "--smoothing_type",
        choices=["gaussian", "bilateral", "median"],
        help="Choose the type of smoothing to apply before processing.",
    )

    parser.add_argument(
        "--smoothing_strength",
        type=int,
        default=5,
        help="Set the strength of the smoothing (kernel size, sigma, etc.).",
    )

    parser.add_argument(
        "--post_process",
        choices=["bilateral", "median", "gaussian_edges", "cartoon", "cartoon2"],
        help="Choose a post-processing effect to apply to the final image.",
    )

    parser.add_argument(
        "--anisotropic_kuwahara",
        action="store_true",
        help="Enable the anisotropic Kuwahara filter.",
    )

    parser.add_argument(
        "--kuwahara_kernel_size",
        type=int,
        default=5,
        help="Kernel size for the anisotropic Kuwahara filter.",
    )

    parser.add_argument(
        "--kuwahara_alpha",
        type=float,
        default=1.0,
        help="Alpha parameter for the anisotropic Kuwahara filter.",
    )

    parser.add_argument(
        "--kuwahara_blur_radius",
        type=int,
        default=2,
        help="Blur radius for the anisotropic Kuwahara filter.",
    )

    parser.add_argument(
        "--kuwahara_zero_crossing",
        type=float,
        default=0.58,
        help="Zero crossing parameter for the anisotropic Kuwahara filter.",
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
        smoothing_type=args.smoothing_type,
        smoothing_strength=args.smoothing_strength,
        post_process=args.post_process,
        anisotropic_kuwahara=args.anisotropic_kuwahara,
        kuwahara_kernel_size=args.kuwahara_kernel_size,
        kuwahara_alpha=args.kuwahara_alpha,
        kuwahara_blur_radius=args.kuwahara_blur_radius,
        kuwahara_zero_crossing=args.kuwahara_zero_crossing,
    )
