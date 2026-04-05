#!/usr/bin/env python

import subprocess
import re
import time
import os



from fractions import Fraction

def parse_value(val_str):
    try:
        if isinstance(val_str, (int, float)):
            return float(val_str)
        if '/' in val_str:
            return float(Fraction(val_str))  # e.g. '1/1000' → 0.001
        return float(val_str)
    except ValueError:
        return None

def filter_values_in_range(data_list, start_raw, end_raw):
    start = parse_value(start_raw)
    end = parse_value(end_raw)
    if start is None or end is None:
        raise ValueError("Invalid start or end range values.")

    # Ensure start <= end for comparison
    start, end = min(start, end), max(start, end)

    result = []
    for val in data_list:
        num = parse_value(val)
        if num is not None and start <= num <= end:
            result.append(val)
    return result






def wait_until_file_accessible(filepath, check_interval=0.5):
    """
    Waits until a file exists and can be opened for read/write access.
    This helps ensure it's not in use by another process (e.g., still being written).
    """
    while True:
        if os.path.exists(filepath):
            try:
                # Try opening the file for read+write to ensure it's not locked
                with open(filepath, 'rb+'):
                    return  # File is ready
            except (OSError, PermissionError):
                pass  # File exists but is still in use
        time.sleep(check_interval)

def wait_for_file(path, check_interval=0.5):
    """Wait until the file exists and is stable."""
    print(f"Waiting for stable file: {path}")
    while wait_until_file_accessible(path):
        time.sleep(check_interval)
    print(f"File ready: {path}")

def rename_with_bash(old_name, new_name):
    """Waits for file to be stable, then renames it using bash mv command."""
    cwd = os.getcwd()
    old_path = os.path.join(cwd, old_name)
    new_path = os.path.join(cwd, new_name)

    wait_for_file(old_path)

    try:
        subprocess.run(["mv", old_path, new_path], check=True)
        print(f"Renamed using bash: '{old_path}' → '{new_path}'")
    except subprocess.CalledProcessError as e:
        print(f"Failed to rename '{old_name}' → '{new_name}': {e}")

def rename_capture_files(new_base_name: str):
    """
    Renames 'capt0000.cr2' to '{new_base_name}.cr2' and
    'capt0001.jpg' to '{new_base_name}.jpg' using bash mv.
    """
    files = {
        "capt0000.cr2": f"{new_base_name}.cr2",
        "capt0001.jpg": f"{new_base_name}.jpg"
    }

    for old_name, new_name in files.items():
        rename_with_bash(old_name, new_name)





def get_camera_settings():
    subprocess.run('lscpu')
    result = subprocess.run(['gphoto2', '--auto-detect'])
    print(result.stdout)
    shutter_speeds = subprocess.check_output(['gphoto2', '--get-config', '/main/capturesettings/shutterspeed']).decode()
    shutter_speeds =  matches = re.findall(r'^Choice:\s+\d+\s+(\S+)$', shutter_speeds, re.MULTILINE)
    shutter_speeds = [v for v in shutter_speeds if re.match(r'^\d+(\.\d+)?$|^\d+/\d+$', v)]
    print("shutter speeds:",shutter_speeds)

    apertures = subprocess.check_output(['gphoto2', '--get-config', '/main/capturesettings/aperture']).decode()
    apertures = re.findall(r'^Choice:\s+\d+\s+(\S+)', apertures, re.MULTILINE)
    apertures = [v for v in apertures if re.match(r'^(\d+(\.\d+)?|\d+/\d+)$', v)]
    print("apertures:", apertures)

    return shutter_speeds, apertures


def shoot_photos(shutter_speed_start, shutter_speed_end, aperture_start, aperture_end):
    shutter_speeds, apertures = get_camera_settings()

    shutter_filtered = filter_values_in_range(shutter_speeds, shutter_speed_start, shutter_speed_end)
    aperture_filtered = filter_values_in_range(apertures, aperture_start, aperture_end)

    print("Filtered shutter speeds:", shutter_filtered)
    print("Filtered apertures:", aperture_filtered)

    for aperture in aperture_filtered:
        subprocess.run(['gphoto2', '--auto-detect', f'--set-config=/main/capturesettings/aperture={aperture}'])
        print(f"aperture set {aperture}")
        time.sleep(1)
        for shutter in shutter_filtered:
            subprocess.run(['gphoto2', '--auto-detect', f'--set-config=/main/capturesettings/shutterspeed={shutter}'])
            print(f"shutterspeed set {shutter_filtered}")
            #time.sleep(1)
            shutter_to_write = shutter.replace("/", "na")
            aperture_to_write = aperture.replace("/", "na")
            filename = f"shutter_{shutter_to_write}_aperture_{aperture_to_write}"
            p = subprocess.run(['gphoto2', '--auto-detect', '--capture-image-and-download', '--force-overwrite'])
            #subprocess.run(['gphoto2', '--auto-detect', '--capture-image-and-download'])
            rename_capture_files(filename)




if __name__ == "__main__":
    import sys

    shutter_speed_start = sys.argv[1]
    shutter_speed_end = sys.argv[2]
    aperture_start = sys.argv[3]
    aperture_end = sys.argv[4]

    shoot_photos(shutter_speed_start, shutter_speed_end, aperture_start, aperture_end)
