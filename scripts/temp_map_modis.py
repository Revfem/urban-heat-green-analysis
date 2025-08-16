import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt


local_path = r"data/MODIS/MOD11A2.A2024193.h19v04.061.2024202041453_MODIS_Grid_8Night_1km_LST.tif"
out_path   = "output/MOD11A2_LST_Night_celsius.tif"

os.makedirs("output", exist_ok=True)


with rasterio.open(local_path) as src:
    raw = src.read(1).astype(np.float32)
    profile = src.profile
    nodata = src.nodata


raw = np.where((raw == nodata) | (raw <= 0), np.nan, raw)

kelvin = raw * 0.02
celsius = kelvin - 273.15


profile.update(dtype=rasterio.float32, nodata=np.nan)
with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(celsius.astype(np.float32), 1)

print("Saved:", out_path)


plt.figure(figsize=(8,6))
im = plt.imshow(celsius, cmap="inferno", vmin=20, vmax=45)  # ajustează vmin/vmax după anotimp
plt.colorbar(im, label="LST Day (°C)")
plt.title("MOD11A2 LST Day (°C)")
plt.show()
