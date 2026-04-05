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
from scipy.ndimage import generic_filter
from scipy.ndimage import convolve


def calculate_e_mop_kof(focal_length_mm, f_number, sensor_pixel_pitch_um, radius, x_offset, y_offset, x_max, y_max):
    tau_mos = 0.9

    focal_length_in_meters = focal_length_mm / 1000
    sensor_pixel_pitch_in_meters = (sensor_pixel_pitch_um / 1000) / 1000

    aperture_diameter = focal_length_in_meters / f_number
    aperture_radius = aperture_diameter / 2
    aperture_area = math.pi * (aperture_radius ** 2)

    cx = x_max // 2 + x_offset
    cy = y_max // 2 + y_offset

    e_mop_of_pixels_kof_1 = []
    e_mop_of_pixels_kof_2 = []
    cos_theta_of_pixels = []

    for y in range(max(0, cy - radius), min(x_max, cy + radius)):
        for x in range(max(0, cx - radius), min(y_max, cx + radius)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                dx = (x - (x_max / 2)) * sensor_pixel_pitch_in_meters
                dy = (y - (y_max / 2)) * sensor_pixel_pitch_in_meters
                dz = focal_length_in_meters
                distance_square = dx ** 2 + dy ** 2 + dz ** 2
                distance = math.sqrt(distance_square)
                cos_theta = dz / distance
                first_part = tau_mos * (focal_length_in_meters ** 2)
                e_mop_of_pixels_kof_1.append((tau_mos * (focal_length_in_meters ** 2)) / distance_square)
                e_mop_of_pixels_kof_2.append((tau_mos * cos_theta) / distance_square)
                cos_theta_of_pixels.append((cos_theta ** 4))

    average_e_mop_1 = sum(e_mop_of_pixels_kof_1) / len(e_mop_of_pixels_kof_1)
    average_cos_tetha = sum(cos_theta_of_pixels) / len(cos_theta_of_pixels)

    print(f"Calculated Emop 1: {average_e_mop_1}")
    print(f"Calculated cos theta^4: {average_cos_tetha}")

    return average_e_mop_1, average_cos_tetha


# === SETTINGS ===
offset_x = 0  # Horizontal offset from image center
offset_y = 0  # Vertical offset from image center
radius = 50  # Circular region radius (in pixels)

sensor_pixel_pitch_um = 5.7

e_mop_calculated = 0
average_e_mop_1 = 0.0
average_cos_tetha = 0.0

# === Automatically use the folder where the script resides ===
script_dir = Path(__file__).resolve().parent
folder_path = script_dir


# === NEW: Matrix to hold XYZ and RGB values by aperture/exposure ===
class XYZRGBMatrix:
    def __init__(self):
        self.apertures = []
        self.exposures = []
        self.raw_green_matrix = []
        self.xyz_matrix = []
        self.rgb_matrix = []
        self.srgb_matrix = []

    def _resize_matrices(self):
        rows = len(self.apertures)
        cols = len(self.exposures)

        def new_matrix():
            return [[None for _ in range(cols)] for _ in range(rows)]

        old_raw_green = self.raw_green_matrix
        old_xyz = self.xyz_matrix
        old_rgb = self.rgb_matrix
        old_srgb = self.srgb_matrix

        self.raw_green_matrix = new_matrix()
        self.xyz_matrix = new_matrix()
        self.rgb_matrix = new_matrix()
        self.srgb_matrix = new_matrix()

        for i, a in enumerate(self.apertures):
            for j, e in enumerate(self.exposures):
                if a in self._old_apertures and e in self._old_exposures:
                    old_i = self._old_apertures.index(a)
                    old_j = self._old_exposures.index(e)
                    self.raw_green_matrix[i][j] = old_raw_green[old_i][old_j]
                    self.xyz_matrix[i][j] = old_xyz[old_i][old_j]
                    self.rgb_matrix[i][j] = old_rgb[old_i][old_j]
                    self.srgb_matrix[i][j] = old_srgb[old_i][old_j]

        self._old_apertures = list(self.apertures)
        self._old_exposures = list(self.exposures)

    def add(self, aperture, exposure, raw_green, xyz, rgb, srgb):
        aperture = float(aperture)
        exposure = float(exposure)

        if exposure not in self.exposures:
            self.exposures.append(exposure)
            self.exposures.sort()

        if aperture not in self.apertures:
            self.apertures.append(aperture)
            self.apertures.sort()

        if not hasattr(self, '_old_apertures'):
            self._old_apertures = []
            self._old_exposures = []

        self._resize_matrices()

        i = self.apertures.index(aperture)
        j = self.exposures.index(exposure)

        self.raw_green_matrix[i][j] = raw_green
        self.xyz_matrix[i][j] = xyz
        self.rgb_matrix[i][j] = rgb
        self.srgb_matrix[i][j] = srgb

    def export_to_csv(self, filename):
        output_rows = [[""] for _ in range(150)]

        # XYZ matrix
        output_rows[0] = ["Aperture \\ Exposure"] + self.exposures
        for i, a in enumerate(self.apertures):
            row = [a]
            for j in range(len(self.exposures)):
                value = self.raw_green_matrix[i][j]
                row.append("" if value is None else f"{value:}")
            output_rows[1 + i] = row

        # EMOP values
        output_rows[16] = ["average_e_mop_1", average_e_mop_1]
        output_rows[17] = ["average_cos_tetha", average_cos_tetha]

        # XYZ matrix (row 20)
        rgb_start = 19
        output_rows[rgb_start] = ["XYZ Matrix"]
        output_rows[rgb_start + 1] = ["Aperture \\ Exposure"] + self.exposures
        for i, a in enumerate(self.apertures):
            row = [a]
            for j in range(len(self.exposures)):
                value = self.xyz_matrix[i][j]
                row.append("" if value is None else f"{value[0]} {value[1]} {value[2]}")
            output_rows[rgb_start + 2 + i] = row

        # RGB matrix (row 40)
        rgb_start = 39
        output_rows[rgb_start] = ["pure sRGB Matrix"]
        output_rows[rgb_start + 1] = ["Aperture \\ Exposure"] + self.exposures
        for i, a in enumerate(self.apertures):
            row = [a]
            for j in range(len(self.exposures)):
                value = self.rgb_matrix[i][j]
                row.append("" if value is None else f"{value[0]} {value[1]} {value[2]}")
            output_rows[rgb_start + 2 + i] = row

        # sRGB matrix (row 60)
        srgb_start = 59
        output_rows[srgb_start] = ["sRGB Matrix"]
        output_rows[srgb_start + 1] = ["Aperture \\ Exposure"] + self.exposures
        for i, a in enumerate(self.apertures):
            row = [a]
            for j in range(len(self.exposures)):
                value = self.srgb_matrix[i][j]
                row.append("" if value is None else f"{value[0]} {value[1]} {value[2]}")
            output_rows[srgb_start + 2 + i] = row

        # Write CSV
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(output_rows)


# === Get EXIF info ===
def get_exif_info(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f, stop_tag="EXIF ExposureTime")

    aperture = float(tags.get("EXIF FNumber", "Not found").values[0])
    exposure = float(tags.get("EXIF ExposureTime", "Not found").values[0])
    focal_length = float(tags.get("EXIF FocalLength", "Not found").values[0])

    return aperture, exposure, focal_length


# === Process a single image and return XYZ + RGB ===
def process_image(file_path):
    print(f"Processing: {file_path}")

    global e_mop_calculated, average_e_mop_1, average_cos_tetha

    aperture, exposure, focal_length = get_exif_info(file_path)

    avg_intensity = 0.0
    sensor_max = 0.0

    # 1. OPEN FILE ONCE
    with rawpy.imread(file_path) as raw:

        # 2. RAW DATA & SATURATION CHECK (FASTEST TO DO FIRST)
        raw_image = raw.raw_image_visible.astype(np.float32)
        h_raw, w_raw = raw_image.shape

        black_level = np.array(raw.black_level_per_channel, dtype=np.float32)
        white_level = float(raw.white_level)
        sensor_max = white_level - np.mean(black_level)

        cx_raw = w_raw // 2 + offset_x
        cy_raw = h_raw // 2 + offset_y

        Y_raw, X_raw = np.ogrid[:h_raw, :w_raw]
        mask_circle_raw = (X_raw - cx_raw) ** 2 + (Y_raw - cy_raw) ** 2 <= radius ** 2

        # === SATURATION CHECK ===
        max_roi_value = np.max(raw_image[mask_circle_raw])
        if max_roi_value >= (white_level * 0.99):
            raise ValueError(f"Saturated channel in ROI. Max: {max_roi_value}, White Level: {white_level}")
        # ========================

        # 3. CALCULATE RAW GREEN AVERAGE
        pattern = raw.raw_pattern.copy()
        cfa_channels = {0: "R", 1: "G", 2: "B", 3: "G"}

        green_mask = np.zeros_like(raw_image, dtype=bool)
        for row in range(2):
            for col in range(2):
                if cfa_channels[pattern[row, col]] == "G":
                    green_mask[row::2, col::2] = True

        mask = green_mask & mask_circle_raw

        bl_map = np.zeros_like(raw_image, dtype=np.float32)
        for row in range(2):
            for col in range(2):
                ch = pattern[row, col]
                bl_map[row::2, col::2] = black_level[ch]

        values = raw_image[mask] - bl_map[mask]
        values = np.clip(values, 0, None)
        norm_values = values / (white_level - black_level.mean())
        avg_intensity = norm_values.mean() if norm_values.size > 0 else np.nan

        # 4. POST-PROCESSING (ONLY HAPPENS IF NOT SATURATED)
        image_rgb = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
            use_camera_wb=False,
            user_wb=[1.0, 1.0, 1.0, 1.0],
            output_color=rawpy.ColorSpace.sRGB,
            no_auto_bright=True,
            gamma=(1.0, 1.0),
            output_bps=16,
            half_size=False,
            use_auto_wb=False
        )

        image_srgb = raw.postprocess(
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=8
        )

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
            use_auto_wb=False,
        )

    # Post-processed images can sometimes have slight border crops compared to RAW
    height_out, width_out, _ = image_xyz.shape

    if e_mop_calculated == 0:
        average_e_mop_1, average_cos_tetha = calculate_e_mop_kof(
            focal_length, aperture, sensor_pixel_pitch_um, radius,
            offset_x, offset_y, width_out, height_out
        )
        e_mop_calculated = 1

    # 5. EXTRACT AVERAGES USING FAST NUMPY MASKS (NO FOR LOOPS)
    cx_out = width_out // 2 + offset_x
    cy_out = height_out // 2 + offset_y

    Y_out, X_out = np.ogrid[:height_out, :width_out]
    mask_circle_out = (X_out - cx_out) ** 2 + (Y_out - cy_out) ** 2 <= radius ** 2

    if not np.any(mask_circle_out):
        raise ValueError("No pixels found in circular region.")

    avg_srgb_pure = np.mean(image_rgb[mask_circle_out], axis=0)
    avg_srgb = np.mean(image_srgb[mask_circle_out], axis=0)
    avg_XYZ = np.mean(image_xyz[mask_circle_out], axis=0)

    # Normalize
    avg_srgb_pure_normalized = avg_srgb_pure.astype(np.float64) / 65535.0
    avg_XYZ_normalized = avg_XYZ.astype(np.float64) / sensor_max

    return {
        "filename": str(file_path),
        "rawG": float(avg_intensity),
        "X": float(avg_XYZ_normalized[0]),
        "Y": float(avg_XYZ_normalized[1]),
        "Z": float(avg_XYZ_normalized[2]),
        "R": float(avg_srgb_pure_normalized[0]),
        "G": float(avg_srgb_pure_normalized[1]),
        "B": float(avg_srgb_pure_normalized[2]),
        "sR": float(avg_srgb[0]),
        "sG": float(avg_srgb[1]),
        "sB": float(avg_srgb[2]),
    }


# === Process all CR2 files ===
matrix = XYZRGBMatrix()

for filename in sorted(os.listdir('.')):
    if filename.lower().endswith(('.raw', '.nef', '.cr2', '.orf', '.rw2')):
        try:
            result = process_image(filename)
            aperture, exposure, _ = get_exif_info(filename)
            rawG = (result['rawG'])
            xyz = (result['X'], result['Y'], result['Z'])
            rgb = (result['R'], result['G'], result['B'])
            srgb = (result['sR'], result['sG'], result['sB'])
            matrix.add(aperture, exposure, rawG, xyz, rgb, srgb)
            print(f"{result['filename']}: Y = {result['Y']:.6f}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if matrix.apertures:
    matrix.export_to_csv(folder_path / "matrix_output.csv")
    print(f"\nMatrix saved to: matrix_output.csv")
else:
    print("\nNo data to save.")

