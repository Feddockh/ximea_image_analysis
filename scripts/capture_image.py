#!/usr/bin/env python3
from ximea import xiapi
import numpy as np
from PIL import Image
import os
import glob
import cv2

def main():
    # ------------- Options -------------
    # Choose which FFC method to use.
    use_builtin_ffc = False         # Uses precomputed FFC files via the camera API.
    compute_ffc_manually = True   # Computes FFC manually using the same TIFF images.
    output_filename = 'python_ffc_applied.jpg' # Output filename for the captured image.
    
    # Ensure that both options are not enabled simultaneously.
    if use_builtin_ffc and compute_ffc_manually:
        print("Both built-in FFC and manual FFC computation options are enabled. Defaulting to auto.")
        compute_ffc_manually = False

    # for i in range(1, 3):

    #     if i == 1:
    #         use_builtin_ffc = True
    #         compute_ffc_manually = False
    #         output_filename = 'python_ffc_auto.jpg'
    #     else:
    #         use_builtin_ffc = False
    #         compute_ffc_manually = True
    #         output_filename = 'python_ffc_manual.jpg'

    # ------------- Camera Setup -------------
    # Create instance for the first connected camera.
    cam = xiapi.Camera()
    print('Opening first camera...')
    cam.open_device()
    
    # Set the exposure (in microseconds).
    cam.set_exposure(100000)
    print('Exposure set to {} us'.format(cam.get_exposure()))

    # Set the gain (in dB).
    cam.set_gain(0)
    print('Gain set to {} dB'.format(cam.get_gain()))

    # Set the image data format to RAW8.
    # This mode returns raw 8-bit sensor data (1 byte per pixel) with no processing.
    cam.set_imgdataformat("XI_RAW8")
    print("Image data format set to XI_RAW8")

    # ------------- Flat Field Correction Setup -------------
    if use_builtin_ffc or compute_ffc_manually:
        # Set the folder for the FFC files (update path as needed).
        # ffc_folder = "~/.local/share/xiCamTool/shading"
        # ffc_folder = os.path.expanduser(ffc_folder)
        ffc_folder = "ffc"

        # Get the list of TIFF files in the folder.
        ffc_files = glob.glob(os.path.join(ffc_folder, "*.tif"))
        if not ffc_files:
            raise FileNotFoundError("Could not find the required FFC files in the folder: {}".format(ffc_folder))
        
        # Sort the files by modification time in descending order.
        ffc_files.sort(key=os.path.getmtime, reverse=True)
        
        # Get the prefix of the most recent FFC files (assumes at least two exist).
        ffc_prefix = os.path.commonprefix(ffc_files[0:2])
        print(f"Found FFC files with prefix: {ffc_prefix}")

        ffc_flat_field_file_name = ffc_prefix + "mid.tif"
        ffc_dark_field_file_name = ffc_prefix + "dark.tif"

    if use_builtin_ffc:
        print("Enabling built-in flat field correction...")

        # Enable flat field correction using the camera’s built-in FFC.
        cam.enable_ffc()
        print("Built-in flat field correction enabled.")

        # Set the flat field and dark field file names.
        cam.set_ffc_flat_field_file_name(ffc_flat_field_file_name)
        print("FFC flat field file name set to:", ffc_flat_field_file_name)
        cam.set_ffc_dark_field_file_name(ffc_dark_field_file_name)
        print("FFC dark field file name set to:", ffc_dark_field_file_name)

    # ------------- Image Acquisition -------------
    # Create an Image instance to store image data and metadata.
    img = xiapi.Image()

    # Start data acquisition.
    print('Starting data acquisition...')
    cam.start_acquisition()

    # Capture a single image.
    print('Capturing image...')
    cam.get_image(img)

    # Retrieve raw image data (as bytes).
    data_raw = img.get_image_data_raw()
    width = img.width
    height = img.height
    print('Captured image dimensions: {} x {}'.format(width, height))

    # --- Convert raw data into a NumPy array ---
    # In RAW8 mode, each pixel is 1 byte, so we expect data_raw to be width*height bytes long.
    expected_length = width * height
    if len(data_raw) != expected_length:
        raise ValueError("Unexpected image data size: expected {} bytes, got {} bytes."
                        .format(expected_length, len(data_raw)))
    
    # Create a 2D NumPy array (grayscale image) from the raw bytes.
    np_image = np.frombuffer(data_raw, dtype=np.uint8).reshape((height, width))

    # ------------- Manual FFC Computation -------------
    if compute_ffc_manually:
        print("Computing flat field correction manually using TIF images...")

        # Load the flat field and dark field images in grayscale.
        flat_field = cv2.imread(ffc_flat_field_file_name, cv2.IMREAD_GRAYSCALE)
        dark_field = cv2.imread(ffc_dark_field_file_name, cv2.IMREAD_GRAYSCALE)
        
        if flat_field is None or dark_field is None:
            raise FileNotFoundError("Could not read flat field or dark field images.")
        
        # Optionally, ensure that the flat/dark field dimensions match the captured image.
        if flat_field.shape != np_image.shape or dark_field.shape != np_image.shape:
            raise ValueError("Flat field/dark field image dimensions do not match the captured image dimensions.")
        
        # Convert images to float32 for computation.
        np_image_float = np_image.astype(np.float32)
        dark_field_float = dark_field.astype(np.float32)
        flat_field_float = flat_field.astype(np.float32)
        
        # Compute the denominator, adding a small epsilon to avoid division by zero.
        epsilon = 1e-6
        ratio = (np_image_float - dark_field_float) / (flat_field_float - dark_field_float + epsilon)
        
        # Scale factor: use the mean of (flat_field - dark_field) to preserve brightness.
        scale = np.mean(flat_field_float - dark_field_float)
        corrected_image = ratio * scale
        corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)
        
        # Replace the raw image with the corrected image.
        np_image = corrected_image
        print("Manual flat field correction complete using TIF images.")

    # ------------- Save the Image -------------
    # Create a PIL image from the (possibly corrected) NumPy array.
    pil_image = Image.fromarray(np_image, mode='L')

    # Save the image as a JPEG file.
    pil_image.save(output_filename, 'JPEG')
    print('Image saved as {}'.format(output_filename))

    # Stop data acquisition and close the camera.
    print('Stopping acquisition...')
    cam.stop_acquisition()
    cam.close_device()
    print('Camera closed.')

if __name__ == '__main__':
    main()
