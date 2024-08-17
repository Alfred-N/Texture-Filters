import cv2
import numpy as np
import argparse

from utils import *
from filters import *
from color_quantization import *


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
    max_iter=1,
    smoothing_type=None,
    smoothing_strength=5,
    post_process=None,
    anisotropic_kuwahara=False,  # New argument to enable anisotropic Kuwahara filter
    variation="ANISOTROPIC",  # Kuwahara filter variation
    use_high_precision=False,  # High precision setting
    uniformity=1,  # Uniformity setting
    sharpness=0.5,  # Sharpness setting
    eccentricity=1.0,  # Eccentricity setting
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
        max_iter=max_iter,
    )
    img_np, nearest_colors_inds_orig = round_img_to_colors(img_np, top_k_colors)
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
            max_iter=max_iter,
        )
        img_np = swap_colors(
            nearest_colors_inds_orig,
            top_k_colors,
            top_k_colors_ref,
            luminance_correction=luminance_correction,
        )

    img_np = downsample_image(img_np, downsample_ratio)

    # Apply Anisotropic Kuwahara Filter if enabled
    if anisotropic_kuwahara:
        img_np = apply_kuwahara_blender(
            img_np,
            variation,
            use_high_precision,
            uniformity,
            sharpness,
            eccentricity,
        )
        if img_np is None:
            return

    if post_process:
        if post_process == "bilateral":
            img_np = apply_bilateral_filter(img_np)
        elif post_process == "median":
            img_np = apply_median_filter(img_np)
        elif post_process == "gaussian_edges":
            img_np = apply_gaussian_blur_and_edges(img_np)
        elif post_process == "cartoon":
            img_np = apply_cartoon_effect(img_np)
        elif post_process == "cartoon2":
            img_np = apply_cartoon_effect_v2(img_np)
        else:
            raise ValueError(f"Unknown post-processing option: {post_process}")

    output_png_path = (
        output_path
        if output_path.endswith(".png")
        else output_path.replace(".exr", ".png")
    )
    cv2.imwrite(output_png_path, img_np)
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
            img_np,
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
        default=25,
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
        default=5,
        help="Number of different random seeds to start from for k-means clustering (increases time linearly).",
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=5,
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
        "--variation",
        type=str,
        default="ANISOTROPIC",
        help="Kuwahara filter variation, default is ANISOTROPIC.",
    )

    parser.add_argument(
        "--use_high_precision",
        action="store_true",
        help="Use high precision mode for the Kuwahara filter.",
    )

    parser.add_argument(
        "--uniformity",
        type=int,
        default=1,
        help="Uniformity of the Kuwahara filter.",
    )

    parser.add_argument(
        "--sharpness",
        type=float,
        default=0.5,
        help="Sharpness of the Kuwahara filter.",
    )

    parser.add_argument(
        "--eccentricity",
        type=float,
        default=1.0,
        help="Eccentricity of the Kuwahara filter.",
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
        max_iter=args.max_iter,
        smoothing_type=args.smoothing_type,
        smoothing_strength=args.smoothing_strength,
        post_process=args.post_process,
        anisotropic_kuwahara=args.anisotropic_kuwahara,
        variation=args.variation,
        use_high_precision=args.use_high_precision,
        uniformity=args.uniformity,
        sharpness=args.sharpness,
        eccentricity=args.eccentricity,
    )
