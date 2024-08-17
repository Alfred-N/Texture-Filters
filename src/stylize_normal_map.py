import cv2
import numpy as np
import platform
import subprocess
import argparse


def convert_exr_to_png(exr_path):
    if platform.system() == "Darwin":  # Check if OS is macOS
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
    if platform.system() == "Darwin":  # Check if OS is macOS
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
    img_np = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img_np is None:
        raise ValueError(f"Failed to load image from {input_path}")

    # If the image has more than 3 channels, select the first 3 (this is common for EXR with alpha)
    if img_np.shape[2] > 3:
        img_np = img_np[:, :, :3]

    return img_np, input_path


def apply_aggressive_bilateral_filter(
    img_np, d=15, sigma_color=100, sigma_space=100, iterations=5
):
    for i in range(iterations):
        img_np = cv2.bilateralFilter(img_np, d, sigma_color, sigma_space)
    return img_np


def apply_aggressive_gaussian_blur(img_np, ksize=25, sigma=10):
    # Ensure the image is in 8-bit format for Gaussian blur
    if img_np.dtype != np.uint8:
        img_np = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX)
        img_np = np.uint8(img_np)

    return cv2.GaussianBlur(img_np, (ksize, ksize), sigma)


def apply_downscale_upscale_to_normal_map(img_np, scale_factor=0.2):
    height, width = img_np.shape[:2]
    small_img = cv2.resize(
        img_np,
        (int(width * scale_factor), int(height * scale_factor)),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.resize(small_img, (width, height), interpolation=cv2.INTER_LINEAR)


def stylize_normal_map(input_path, output_path, method="bilateral"):
    # Load the normal map
    img_np, converted_input_path = load_image(input_path)
    input_is_exr = input_path.endswith(".exr")

    # Apply the chosen stylization method
    if method == "bilateral":
        img_stylized = apply_aggressive_bilateral_filter(img_np)
    elif method == "gaussian":
        img_stylized = apply_aggressive_gaussian_blur(img_np)
    elif method == "downscale_upscale":
        img_stylized = apply_downscale_upscale_to_normal_map(img_np)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Save the stylized normal map
    output_png_path = (
        output_path
        if output_path.endswith(".png")
        else output_path.replace(".exr", ".png")
    )
    cv2.imwrite(output_png_path, img_stylized)
    print(f"Stylized normal map saved to {output_png_path}")

    # Convert back to EXR if necessary
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
        "-o",
        "--output_path",
        type=str,
        required=True,
        help="Path to save the stylized normal map.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["bilateral", "gaussian", "downscale_upscale"],
        default="bilateral",
        help="Method to use for stylization: 'bilateral', 'gaussian', or 'downscale_upscale'.",
    )

    args = parser.parse_args()

    stylize_normal_map(args.input_path, args.output_path, method=args.method)
