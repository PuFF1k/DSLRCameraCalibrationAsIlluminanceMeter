# Camera-Based Illuminance Measurement

A Python toolkit for using digital cameras as measurement devices to analyze and regulate external lighting quality.

## Requirements
System: `gphoto2` (for tethered capture)
Python: `pip install Pillow numpy exifread rawpy colour-science opencv-python matplotlib cmcrameri scipy`

## Scripts

* **TakePhotos.py**: Automates tethered camera captures across user-defined exposure and aperture brackets.
* **ParseCalibrationPhotos.py**: Extracts raw sensor data from calibration images to generate a response matrix (`matrix_output.csv`).
* **ParseValidationPhotos.py**: Extracts luminance (Y channel) at predefined coordinates to validate linearity and flag sensor clipping.
* **CreateIlluminanceHeatmap.py**: Applies optical falloff and radiometric corrections to raw images to generate spatial illuminance heatmaps.

## Workflow

1.  **Capture**: Run `TakePhotos.py` to gather bracketed images.
2.  **Calibrate**: Run `ParseCalibrationPhotos.py` to establish sensor baselines.
3.  **Validate**: Run `ParseValidationPhotos.py` on target scenes.
4.  **Visualize**: Run `CreateIlluminanceHeatmap.py` to output the final data heatmaps.
