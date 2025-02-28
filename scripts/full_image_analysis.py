#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import argparse

from utils import demosaic_ximea_5x5

def plot_multiple_spectral_intensities(image_spectra, spectral_range):
    """
    Plot the spectral intensities for multiple images on the same graph.
    
    Parameters:
      image_spectra : dict
          Keys are image filenames and values are 1D numpy arrays containing
          the average intensity for each spectral band.
      spectral_range : numpy.ndarray
          1D array of wavelength values (in nm) corresponding to each band.
    """
    plt.figure(figsize=(10, 5))
    x = np.arange(len(spectral_range))
    
    # Loop over each image and plot its spectral intensities.
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    color_index = 0
    for filename, intensities in image_spectra.items():
        # Create a smooth curve for visualization using spline interpolation.
        x_smooth = np.linspace(x.min(), x.max(), 300)
        intensities_smooth = make_interp_spline(x, intensities)(x_smooth)
        plt.plot(x_smooth, intensities_smooth, linewidth=2, label=filename, color=colors[color_index % len(colors)])
        plt.plot(x, intensities, 'o', color=colors[color_index % len(colors)])  # Plot original points
        color_index += 1
        
    plt.xticks(x, [f"{int(wl)} nm" for wl in spectral_range], rotation=45)
    plt.xlabel("Band Index")
    plt.ylabel("Average Intensity")
    plt.title("Spectral Bands")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    
    image_files = ["ros_ffc_applied.jpg", "python_ffc_applied.jpg"]

    image_spectra = {}
    spectral_range = []
    for image_file in image_files:
        print(f"Processing image: {image_file}")

        # Convert the image to a multispectral hypercube.
        # The hypercube should be a dict of 2D numpy arrays, where each key is a spectral band.
        hypercube = demosaic_ximea_5x5(image_file)
        
        # Compute the average intensity of each spectral band.
        avg_intensities = np.array([np.mean(hypercube[band]) for band in hypercube])

        # Retrieve the spectral range from the first image.
        if len(spectral_range) == 0:
            spectral_range = np.array(list(hypercube.keys()))

        # Store the average intensities for this image.
        image_spectra[image_file] = avg_intensities
        print(f"Average intensities for {image_file}: {avg_intensities}\n")
    
    # Plot all images' spectral intensities on a single graph.
    plot_multiple_spectral_intensities(image_spectra, spectral_range)

if __name__ == "__main__":
    main()
