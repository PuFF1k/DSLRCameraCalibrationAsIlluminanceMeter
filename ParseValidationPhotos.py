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




# === SETTINGS ===

sensor_pixel_pitch_um = 5.7

# === Automatically use the folder where the script resides ===
script_dir = Path(__file__).resolve().parent
folder_path = script_dir


def process_regions(file_path, X, Y, focal_length_mm, f_number, exposure, sensor_pixel_pitch_um, radius=22):
    """
    Processes circular regions around (X[i,j], Y[i,j]) and returns a 2D array of Y values from XYZ.
    """
    print(f"Processing {file_path}")

    focal_length_in_meters = focal_length_mm / 1000

    with rawpy.imread(file_path) as raw:
        # 1. Grab raw sensor data just for the clipping check
        raw_image = raw.raw_image_visible.astype(np.float32)
        white_level = float(raw.white_level)
        h_raw, w_raw = raw_image.shape

        # 2. Process the image to XYZ 16-bit
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

        # 3. Handle Canon RAW orientation flip to match raw.raw_image_visible geometry
        flip = raw.sizes.flip
        if flip == 6:
            image_xyz = np.rot90(image_xyz, 1)
        elif flip == 5:
            image_xyz = np.rot90(image_xyz, -1)

        # Extract the Y channel (channel 1 in XYZ) and normalize
        Y_channel = image_xyz[:, :, 1]
        black_level = np.array(raw.black_level_per_channel, dtype=np.float32)
        white_level = float(raw.white_level)
        sensor_max = white_level - np.mean(black_level)
        Y_norm = np.clip(Y_channel / sensor_max, 0, 1)

        # Prepare output matrix (dtype=object supports mixing floats and "!float" strings)
        pixelValuesXYZ = np.zeros_like(X, dtype=object)

        # Iterate over coordinates
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                cx, cy = X[i, j], Y[i, j]

                # Bounding box of the circle, clipped to image boundaries
                x_min = max(int(cx - radius), 0)
                x_max = min(int(cx + radius + 1), w_raw)
                y_min = max(int(cy - radius), 0)
                y_max = min(int(cy + radius + 1), h_raw)

                # Local coordinates inside the subregion
                yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
                dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
                mask = dist2 <= radius ** 2

                # === Saturation Check ===
                sub_raw = raw_image[y_min:y_max, x_min:x_max]
                region_raw_values = sub_raw[mask]

                is_clipped = False
                if region_raw_values.size > 0 and np.max(region_raw_values) >= (white_level * 0.99):
                    is_clipped = True

                # === Y Channel Average ===
                sub_Y = Y_norm[y_min:y_max, x_min:x_max]
                region_Y_values = sub_Y[mask]

                if region_Y_values.size == 0:
                    pixelValuesXYZ[i, j] = 0
                    continue

                avg_Y = max(np.mean(region_Y_values), 0)

                # Assign value, prepending "!" if any raw pixel in this region was clipped
                pixelValuesXYZ[i, j] = f"!{avg_Y}" if is_clipped else avg_Y

    # Save to CSV
    csv_saver = RegionResultsCSV("cardboardAnalysys.csv")
    # Notice we removed 'data' (the old green pixels variable) from the arguments
    csv_saver.save_results(file_path, exposure, f_number, focal_length_in_meters, pixelValuesXYZ)



def get_exif_info(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f, stop_tag="EXIF ExposureTime")

    aperture = float(tags.get("EXIF FNumber", "Not found").values[0])
    exposure = float(tags.get("EXIF ExposureTime", "Not found").values[0])
    focal_length = float(tags.get("EXIF FocalLength", "Not found").values[0])

    return aperture, exposure, focal_length


class RegionResultsCSV:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    # Removed the extra data parameter
    def save_results(self, file_name, exposure, aperture, focalLength, data_xyz):
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["File", "Exposure", "Aperture", "Focal length"])
            writer.writerow([file_name, exposure, aperture, focalLength])

            # Write only the Y data
            writer.writerow(["Y from XYZ"])
            for row in data_xyz:
                writer.writerow(row)

            writer.writerow([])  # blank line between entries
            writer.writerow([])  # blank line between entries




# === Define X and Y matrices ===
X = np.array([
    [108.221652129297, 501.06396701908, 897.100121867316, 1289.9424367571, 1682.78475164688, 2075.62706653666, 2463.14631482902, 2855.98862971881, 3248.83094460859, 3640.60864617889, 4033.45096106867],
    [112.229861467419, 500.813723079264, 896.8498779275, 1292.88603277574, 1684.66373434603, 2070.05375599943, 2465.02529752818, 2858.93222573744, 3249.64531398826, 3638.2291756001, 4034.26533044834],
    [114.23396613648, 506.011667706778, 899.918595916046, 1294.8901374448, 1684.53861237613, 2072.05786066849, 2463.83556223878, 2856.67787712857, 3248.45557869886, 3637.03944031071, 4034.14020847843],
    [114.23396613648, 508.140894345747, 900.98320923553, 1294.8901374448, 1687.73245233458, 2072.05786066849, 2464.90017555827, 2858.80710376754, 3248.45557869886, 3638.10405363019, 4035.20482179791],
    [116.238070805541, 509.080385695324, 898.728860626653, 1293.7004021554, 1687.60733036467, 2072.99735201806, 2463.71044026888, 2857.61736847814, 3247.26584340947, 3636.9143183408, 4036.14431314749],
    [118.242175474602, 510.019877044901, 899.668351976229, 1296.76912014395, 1687.48220839476, 2073.93684336764, 2464.64993161845, 2858.55685982772, 3249.26994807853, 3636.78919637089, 4036.01919117758],
    [117.240123140072, 504.759371432432, 896.53707300273, 1294.70245448994, 1685.41554274075, 2076.12863099156, 2466.84171924238, 2858.61942081267, 3251.46173570246, 3637.91637067533, 4035.01713884305],
    [117.240123140072, 506.888598071401, 900.795526280668, 1293.63784117045, 1685.41554274075, 2077.19324431105, 2464.71249260341, 2857.55480749319, 3247.20328242452, 3638.98098399482, 4036.08175216254]
])

Y = np.array([
    [-72.8247155606242, -69.6275872485232, -68.5618778111562, -68.5618778111562, -67.4961683737892, -67.4961683737892, -68.5618778111562, -69.6275872485232, -69.6275872485232, -71.7590061232572, -73.8904249979912],
    [320.221450019784, 320.860444608724, 322.879353305104, 324.572394398916, 324.898262001484, 324.221045563959, 323.895177961391, 323.230702140061, 324.585135015112, 322.240358716164, 321.250015292266],
    [713.000126522834, 714.734558653596, 713.188002714658, 717.012462683204, 715.465906744266, 715.465906744266, 714.462822704173, 717.284198632627, 719.833838611658, 717.827670531473, 718.371142430319],
    [1103.90637948438, 1083.09211623153, 1090.05107911491, 1113.87459448803, 1102.96592084156, 1097.01004199828, 1101.96283680147, 1124.7832681345, 1118.82738929122, 1128.73297889759, 1138.63856850397],
    [1496.08745243544, 1523.35859770976, 1532.44897946786, 1500.09978859581, 1501.1028726359, 1509.19017035391, 1500.09978859581, 1466.74751368365, 1474.83481140167, 1464.74134560347, 1454.64787980527],
    [1895.081003115, 1904.90273296377, 1905.90581700386, 1892.28086154687, 1893.28394558696, 1893.28394558696, 1882.06214495412, 1877.65282204978, 1881.05906091403, 1868.83417624109, 1866.82800816091],
    [2290.32970671154, 2300.74904011231, 2296.81206996541, 2286.93196159142, 2285.46501853802, 2282.99499144452, 2274.58182612394, 2276.04876917734, 2268.63868789685, 2269.10254691017, 2264.62635173648],
    [2678.83705172099, 2675.63992340889, 2674.57421397153, 2672.50542049407, 2671.4397110567, 2669.37091757924, 2664.23024658423, 2663.22716254414, 2661.15836906668, 2657.08340750903, 2650.93965247393]
])

# === Example usage ===

for filename in sorted(os.listdir('.')):
    if filename.lower().endswith(('.raw', '.nef', '.cr2', '.orf', '.rw2')):
        try:
            aperture, exposure, focal_length = get_exif_info(filename)

            # Process the regions defined by X and Y
            process_regions(
                filename,
                X, Y,
                focal_length_mm=focal_length,
                f_number=aperture,
                exposure=exposure,
                sensor_pixel_pitch_um=sensor_pixel_pitch_um  # <-- replace with your sensor's pixel pitch (µm)
            )
        except Exception as e:
            print(f"Error processing {filename}: {e}")




print("Processing finished, results saved to cardboardAnalysys.csv")
