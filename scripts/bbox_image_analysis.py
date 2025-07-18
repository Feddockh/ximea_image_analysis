# Hayden Feddock
# 1/29/2025

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy.interpolate import make_interp_spline

from utils import demosaic_ximea_5x5, hypercube_dict_to_array

class ImageBoxSelector:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.fig, self.ax = plt.subplots()
        self.rect_selector = None
        self.box_coords = None

    def on_select(self, eclick, erelease):
        """
        Event handler for rectangle selection.
        eclick: Mouse click event (start point)
        erelease: Mouse release event (end point)
        """
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.box_coords = (x1, y1, x2, y2)
        print(f"Box selected: {self.box_coords}")

    def display_image(self):
        """
        Display the image and allow the user to select a bounding box.
        """
        self.ax.imshow(self.image)
        self.ax.set_title("Draw a box and close the window when done")

        # Create RectangleSelector widget
        self.rect_selector = RectangleSelector(
            self.ax,
            onselect=self.on_select,
            useblit=True,
            props=dict(facecolor='red', edgecolor='black', alpha=0.3, fill=True),
            interactive=True
        )

        plt.show()

    def get_box(self):
        """
        Return the selected box coordinates (x1, y1, x2, y2) after the image window is closed.
        """
        return self.box_coords
    
def plot_spectral_intensities(band_intensities, spectral_range, image_name=""):
    """
    Plots the spectral bands of a multispectral image.
    
    Parameters:
    - band_intensities: numpy.ndarray of shape (height, width, num_bands),
    - spectral_range: numpy.ndarray of shape (num_bands,),
      where each value corresponds to the wavelength of a spectral band.
    - image_name: str, name of the image (default: ""). Used for the plot title.
    """

    # Plot the spectral bands with curving lines
    x = np.arange(len(band_intensities))
    x_smooth = np.linspace(x.min(), x.max(), 300)
    band_intensities_smooth = make_interp_spline(x, band_intensities)(x_smooth)

    if not image_name:
        image_name = "Spectral Bands"
    else:
        image_name = f"Spectral Bands ({image_name})"
    
    # Create a high quality figure
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(x_smooth, band_intensities_smooth, linewidth=2)
    plt.plot(x, band_intensities, 'o', color='red')
    plt.xticks(x, [f"{int(wavelength)} nm" for wavelength in spectral_range], rotation=45)
    plt.xlabel("Band Index")
    plt.ylabel("Average Intensity")
    plt.title("Spectral Bands")
    plt.grid()
    
    # Set y-axis limits from 0 to 255 intensity values
    plt.ylim(0, 255)
    
    plt.show()

if __name__ == "__main__":

    # Bands and pattern size
    pattern_size = 5
    spectral_bands = 25
    spectral_range = np.linspace(665, 960, spectral_bands)

    # Initialize the band intensities
    folder = "fb_images"
    image_names = os.listdir("fb_images")
    # image_names = ["ros image.jpg"]
    band_intensities = np.zeros((len(image_names), spectral_bands))

    # Load the image paths
    for i, image_name in enumerate(image_names):
        image_path = os.path.join(folder, image_name)

        selector = ImageBoxSelector(image_path)
        selector.display_image()
        box = selector.get_box()
        
        if not box or box[0] == box[2] or box[1] == box[3]:
            print("No box selected.")
            continue
        print(f"Final selected box: {box}")

        # Convert the image to a hypercube
        hypercube_dict = demosaic_ximea_5x5(image_path) # TODO: Fix this function
        hypercube = hypercube_dict_to_array(hypercube_dict)

        # Extract the selected box from the hypercube
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = x1 // pattern_size, y1 // pattern_size, x2 // pattern_size, y2 // pattern_size
        selected_box = hypercube[:, y1:y2, x1:x2]
        print(f"Selected box shape: {selected_box.shape}")

        # Compute the average intensity of each spectral band
        band_intensities[i] = np.mean(selected_box, axis=(1, 2))

    # Compute the average intensity across all images
    avg_band_intensities = np.mean(band_intensities, axis=0)

    # Plot the spectral intensities
    plot_spectral_intensities(avg_band_intensities, spectral_range, image_name="Average Spectral Intensities")









