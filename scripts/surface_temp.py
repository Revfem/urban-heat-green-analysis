import rasterio
import numpy as np
import matplotlib.pyplot as plt

st_path = "data/LANDSAT8/LC08_L2SP_182029_20240811/LC08_L2SP_182029_20240811_20240815_02_T1_ST_B10.TIF"

with rasterio.open(st_path) as src:
    band10 = src.read(1)
    profile = src.profile

scale_factor = 0.00341802
offset = 149.0

kelvin = band10*scale_factor + offset
celsius = kelvin - 273.15


profile.update(dtype=rasterio.float32)

with rasterio.open("output/temperature_celius.tif","w",**profile) as dst:
    dst.write(celsius.astype(np.float32),1)

plt.figure(figsize=(8, 6))
temp_plot = plt.imshow(celsius, cmap='inferno')
plt.colorbar(temp_plot, label="Temperature (°C)")
plt.title("Surface Temperature from Landsat 8 - Band 10")
plt.show()