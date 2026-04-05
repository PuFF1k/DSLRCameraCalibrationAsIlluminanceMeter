#!/usr/bin/env python
import os
from PIL import Image
import numpy as np
import exifread
import rawpy
import colour
from pathlib import Path
import csv
import math
import cv2
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
from scipy.ndimage import generic_filter
from scipy.ndimage import convolve


script_dir = Path(__file__).resolve().parent
folder_path = script_dir

def get_exif_info(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f, stop_tag="EXIF ExposureTime")

    aperture = float(tags.get("EXIF FNumber", "Not found").values[0])
    exposure = float(tags.get("EXIF ExposureTime", "Not found").values[0])
    focal_length = float(tags.get("EXIF FocalLength", "Not found").values[0])

    return aperture, exposure, focal_length

def create_luminance_heatmap(file_path, output_path="heatmap.png"):
    aperture, exposure, focal_length = get_exif_info(filename)

    with rawpy.imread(file_path) as raw:
        image_xyz = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
            use_camera_wb=False,
            user_wb=[1.0, 1.0, 1.0, 1.0],
            output_color=rawpy.ColorSpace.XYZ,
            no_auto_bright=True,
            no_auto_scale=True,
            gamma=(1.0, 1.0),
            output_bps=16,
            half_size=False,
            use_auto_wb=False
        ).astype(np.float32)

        flip = raw.sizes.flip

        # Canon RAW orientation correction to match raw.raw_image_visible
        if flip == 6:
            # rawpy rotated 90° CW → undo by 90° CCW
            image_xyz = np.rot90(image_xyz, 1)
        elif flip == 5:
            # rawpy rotated 90° CCW → undo by 90° CW
            image_xyz = np.rot90(image_xyz, -1)
        # leave all other orientations unchanged

        # Extract XYZ channels (RAW → XYZ already done)
        X = image_xyz[:, :, 0]
        Y = image_xyz[:, :, 1]  # LUMINANCE CHANNEL
        Z = image_xyz[:, :, 2]

        h, w = Y.shape
        # Hardcoded optical center position (in pixels)
        cx, cy = 2048, 1365

        # Compute radius for each pixel
        yy, xx = np.indices((h, w))
        r = np.sqrt((xx - cx)**2 + (yy - cy)**2)

        # Compute angle theta for each pixel
        sensor_pixel_pitch = sensor_pixel_pitch = (5.7 / 1000) / 1000
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        dx = (xx - cx) * sensor_pixel_pitch
        dy = (yy - cy) * sensor_pixel_pitch
        dz = focal_length/1000
        cos_theta4 = (dz / np.sqrt(dx*dx + dy*dy + dz*dz)) ** 4

        t = exposure
        tau = 0.9
        N = aperture
        black_level = np.array(raw.black_level_per_channel, dtype=np.float32)
        white_level = float(raw.white_level)
        sensor_max = white_level - np.mean(black_level)

        # Apply heatmap formula on normalized values
        Y_norm = np.clip(Y / sensor_max, 0, 1)

        # To create Y heatmap
        #heatmap_data = Y_norm

        # To create Illuminance heatmap
        heatmap_data = (((Y_norm /0.924388899447783) ** (1/1.00367523487035)) * ((4 * (N**2)) / (tau * math.pi * cos_theta4 * t))) * (math.pi / 0.28)

        heatmap_data = np.clip(heatmap_data, 0, None)

        # --- ROI definition ---
        roi_x = 127
        roi_y = 14
        roi_width = 3675
        roi_height = 2567

        heatmap_data = heatmap_data[
                       roi_y: roi_y + roi_height,
                       roi_x: roi_x + roi_width
                       ]

        h, w = heatmap_data.shape

        # Determine absolute min/max for scaling
        vmin = float(np.min(heatmap_data))
        vmax = float(np.max(heatmap_data))

        # Plot heatmap with colorbar (pixel-perfect)
        dpi = 100
        fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
        
        # Switched from "gist_rainbow" to cmc.batlowK
        im = ax.imshow(heatmap_data, cmap=cmc.batlowK, origin="upper", vmin=vmin, vmax=vmax)
        ax.axis("off")

        # Add colorbar with exact limits
        cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cbar.set_label("Heatmap Value")
        cbar.set_ticks(np.linspace(vmin, vmax, 6))  # 6 evenly spaced ticks

        # Save PNG (original resolution + scale bar on the right)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        print(f"Heatmap with scale saved to {output_path} ({w}x{h} + scale bar)")

        # Pixel-perfect export (1 array value = 1 image pixel)

        pixel_output_path = Path(output_path).with_name("heatmap_pixel_exact.png")

        h_pp, w_pp = heatmap_data.shape
        dpi_pp = 100

        fig_pp = plt.figure(figsize=(w_pp / dpi_pp, h_pp / dpi_pp), dpi=dpi_pp)
        ax_pp = fig_pp.add_axes([0, 0, 1, 1])

        # Switched from "inferno" to cmc.batlowK
        ax_pp.imshow(
            heatmap_data,
            cmap=cmc.batlowK,
            origin="upper",
            interpolation="nearest"
        )

        ax_pp.axis("off")

        fig_pp.savefig(
            pixel_output_path,
            dpi=dpi_pp,
            bbox_inches=None,
            pad_inches=0
        )

        plt.close(fig_pp)

        print(f"Pixel-perfect heatmap saved to {pixel_output_path} ({w_pp}x{h_pp})")


for filename in sorted(os.listdir('.')):
    if filename.lower().endswith(('.raw', '.nef', '.cr2', '.orf', '.rw2')):
        try:
            create_luminance_heatmap(filename)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Processing finished")
