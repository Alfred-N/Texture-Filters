import cv2
import platform
import subprocess
import os
import shutil
import tempfile


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


def apply_kuwahara_blender(
    img_np, variation, use_high_precision, uniformity, sharpness, eccentricity
):
    """Apply the anisotropic Kuwahara filter using Blender."""
    print("Applying Anisotropic Kuwahara Filter ...")

    # Create a temporary directory for Blender processing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_image_path = os.path.join(temp_dir, "input_temp.png")
        output_image_path = os.path.join(temp_dir, "input_temp_kuwahara.png")

        # Save the current img_np as a temporary PNG file
        cv2.imwrite(input_image_path, img_np)

        # Check if Blender is in PATH
        blender_executable = shutil.which("blender")
        if blender_executable is None:
            os_type = platform.system()
            print(
                "Blender is not found in your PATH. Please add Blender to your PATH environment variable."
            )
            if os_type == "Darwin":  # macOS
                print("For macOS, add the following to your .zshrc or .bash_profile:")
                print('export PATH="/Applications/Blender.app/Contents/MacOS:$PATH"')
            elif os_type == "Windows":  # Windows
                print(
                    "For Windows, add Blender's directory to your PATH via System Properties."
                )
            elif os_type == "Linux":  # Linux
                print("For Linux, add the following to your ~/.bashrc or ~/.zshrc:")
                print('export PATH="/path/to/blender:$PATH"')
                print(
                    "Replace '/path/to/blender' with the actual path to Blender's executable."
                )
            return None

        # Construct the Blender command
        blender_command = [
            blender_executable,
            "--background",
            "--python",
            "src/blender_kuwahara.py",
            "--",
            input_image_path,
            "--variation",
            variation,
            "--uniformity",
            str(uniformity),
            "--sharpness",
            str(sharpness),
            "--eccentricity",
            str(eccentricity),
        ]

        if use_high_precision:
            blender_command.append("--use_high_precision")

        # Run Blender to apply the filter
        try:
            subprocess.run(blender_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Blender process failed with error: {e}")
            return None

        # Load the processed image from Blender
        img_np = cv2.imread(output_image_path)
        if img_np is None:
            print(f"Failed to load the output image from Blender: {output_image_path}")
            return None

        return img_np
