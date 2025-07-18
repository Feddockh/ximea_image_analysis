import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from utils import demosaic_ximea_5x5, hypercube_dict_to_array
from bbox_image_analysis import ImageBoxSelector


def collect_boxes(label):
    all_band_intensities = []
    print(f"\n🟩 Starting {label.upper()} box selection...\n")
    for i, image_name in enumerate(image_names):
        print(f"[{i+1}/{len(image_names)}] Processing image: {image_name}")
        image_path = os.path.join(folder, image_name)

        selector = ImageBoxSelector(image_path)
        selector.display_image()
        box = selector.get_box()

        if not box or box[0] == box[2] or box[1] == box[3]:
            print("⚠️ No box selected. Skipping.\n")
            continue

        print(f"✅ {label.capitalize()} box selected: {box}\n")

        hypercube_dict = demosaic_ximea_5x5(image_path)
        hypercube = hypercube_dict_to_array(hypercube_dict)

        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = x1 // pattern_size, y1 // pattern_size, x2 // pattern_size, y2 // pattern_size
        selected_box = hypercube[:, y1:y2, x1:x2]

        band_avg = np.mean(selected_box, axis=(1, 2))
        all_band_intensities.append(band_avg)

    return np.array(all_band_intensities)


if __name__ == "__main__":

    pattern_size = 5
    spectral_bands = 25
    spectral_range = np.linspace(665, 960, spectral_bands)

    folder = "fb_images"
    image_names = os.listdir(folder)

    positives = []
    negatives = []

    # Step 1: collect positives
    positives = collect_boxes("positive")

    # Step 2: collect negatives
    negatives = collect_boxes("negative")

    # Save results for future plotting
    np.savez("spectral_profiles.npz",
            positives=positives,
            negatives=negatives,
            spectral_range=spectral_range)
    print("✅ Saved spectral data to 'spectral_profiles.npz'")

    # Step 3: plot both
    x = np.arange(spectral_bands)
    x_smooth = np.linspace(x.min(), x.max(), 300)

    pos_avg = np.mean(positives, axis=0)
    neg_avg = np.mean(negatives, axis=0)

    pos_smooth = make_interp_spline(x, pos_avg)(x_smooth)
    neg_smooth = make_interp_spline(x, neg_avg)(x_smooth)

    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(x_smooth, pos_smooth, label="Positive", color="red", linewidth=2)
    plt.plot(x_smooth, neg_smooth, label="Negative", color="green", linewidth=2)
    plt.scatter(x, pos_avg, color="red")
    plt.scatter(x, neg_avg, color="green")
    plt.xticks(x, [f"{int(w)} nm" for w in spectral_range], rotation=45)
    plt.xlabel("Wavelength")
    plt.ylabel("Average Intensity")
    plt.title("Spectral Signature: Positive vs Negative")
    plt.grid()
    plt.legend()
    plt.ylim(0, 255)
    plt.tight_layout()
    plt.show()
