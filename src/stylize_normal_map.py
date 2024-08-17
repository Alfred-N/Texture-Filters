import numpy as np
import os
import argparse
import cv2
from utils import load_image, convert_exr_to_png, convert_png_to_exr
from normal_map_fns.detect_flat import detect_flat_areas_by_variance
from normal_map_fns.interpolation import apply_linear_smoothing


def apply_mask(original_img, smoothed_img, mask):
    """Applies the mask to preserve flat areas in the original image."""
    combined_img = np.where(mask[..., None], original_img, smoothed_img)
    return combined_img


def stylize_normal_map(input_path, method="linear", preserve_flat_areas=False):
    img_np, converted_input_path = load_image(input_path)
    input_is_exr = input_path.endswith(".exr")

    # Step 2: Detect flat areas if requested using neighborhood variance
    flat_areas_mask = None
    if preserve_flat_areas:
        flat_areas_mask = detect_flat_areas_by_variance(img_np)
        flat_areas_mask = ~flat_areas_mask

    # Step 3: Apply smoothing to a copy of the image
    smoothed_img = apply_linear_smoothing(img_np)

    # Step 4: Combine the original and smoothed images using the mask
    if flat_areas_mask is not None:
        combined_img = apply_mask(img_np, smoothed_img, flat_areas_mask)
    else:
        combined_img = smoothed_img

    # Step 5: Save the result
    base_name, ext = os.path.splitext(input_path)
    output_path = f"{base_name}_slerp{ext}"

    output_png_path = (
        output_path
        if output_path.endswith(".png")
        else output_path.replace(".exr", ".png")
    )
    cv2.imwrite(output_png_path, combined_img)
    print(f"Stylized normal map saved to {output_png_path}")

    if input_is_exr:
        output_path = convert_png_to_exr(output_png_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stylize a normal map by reducing high-frequency details."
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        required=True,
        help="Path to the input normal map.",
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["linear"],
        default="linear",
        help="Method to use for stylization: 'linear'.",
    )

    parser.add_argument(
        "--preserve-rough-areas",
        action="store_true",
        help="Preserve rough areas (non-blue regions) in the normal map during smoothing.",
    )

    args = parser.parse_args()

    stylize_normal_map(
        args.input_path,
        method=args.method,
        preserve_flat_areas=args.preserve_flat_areas,
    )
