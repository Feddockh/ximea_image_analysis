import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Load saved data
data = np.load("spectral_profiles.npz")
positives = data["positives"]
negatives = data["negatives"]
spectral_range = data["spectral_range"]

x = np.arange(len(spectral_range))
x_smooth = np.linspace(x.min(), x.max(), 300)

pos_avg = np.mean(positives, axis=0)
neg_avg = np.mean(negatives, axis=0)

pos_smooth = make_interp_spline(x, pos_avg)(x_smooth)
neg_smooth = make_interp_spline(x, neg_avg)(x_smooth)

# Plot
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
# plt.ylim(0, 255)
plt.tight_layout()
plt.show()
