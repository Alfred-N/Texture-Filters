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


def downsample_image(img_np, downsample_ratio):
    # Calculate the new dimensions of the image
    new_width = int(img_np.shape[1] * downsample_ratio)
    new_height = int(img_np.shape[0] * downsample_ratio)
    return cv2.resize(img_np, (new_width, new_height))


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
