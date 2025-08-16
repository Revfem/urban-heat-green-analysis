import os
import gdown
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import shutil

shutil.rmtree("data", ignore_errors=True)
# link fisier - trb schimbat individual B10!!
file_url = "https://drive.google.com/file/d/156ugihit9fWZLt78WhVZKRfbSYsHcbHD/view?usp=drive_link"


local_path = "data/LANDSAT8/LC08_L2SP_182029_20240811_ST_B10.TIF"
os.makedirs(os.path.dirname(local_path), exist_ok=True)


if not os.path.exists(local_path):
    gdown.download(url=file_url, output=local_path, quiet=False, fuzzy=True)


scale_factor = 0.00341802
offset = 149.0

with rasterio.open(local_path) as src:
    band10 = src.read(1)
    profile = src.profile

kelvin = band10 * scale_factor + offset
celsius = kelvin - 273.15

profile.update(dtype=rasterio.float32)

os.makedirs("output", exist_ok=True)
out_path = "output/temperature_celsius.tif"
with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(celsius.astype(np.float32), 1)

plt.figure(figsize=(8, 6))
temp_plot = plt.imshow(celsius, cmap='inferno')
plt.colorbar(temp_plot, label="Temperature (°C)")
plt.title("Surface Temperature from Landsat 8 - Band 10")
plt.show()
