import numpy as np
import rasterio
import matplotlib.pyplot as plt

b4_path = "data/LANDSAT8/LC08_L2SP_182029_20240811/LC08_L2SP_182029_20240811_20240815_02_T1_SR_B4.TIF" 
b5_path = "data/LANDSAT8/LC08_L2SP_182029_20240811/LC08_L2SP_182029_20240811_20240815_02_T1_SR_B5.TIF"  

with rasterio.open(b5_path) as s_nir:
    nir = s_nir.read(1).astype(np.float32)
    profile = s_nir.profile 

with rasterio.open(b4_path) as s_red:
    red = s_red.read(1).astype(np.float32)


scale = 0.0000275
offset = -0.2

nir = nir * scale + offset
red = red * scale + offset

# # opțional: limitează la [0, 1] ca să eviți valori aberante după offset
# nir = np.clip(nir, 0, 1)
# red = np.clip(red, 0, 1)

ndvi = (nir - red) / (nir + red + 1e-6)

profile.update(dtype=rasterio.float32)
with rasterio.open("output/ndvi_l8.tif", "w", **profile) as dst:
    dst.write(ndvi.astype(np.float32), 1)

plt.figure(figsize=(8, 6))
im = plt.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
plt.colorbar(im, label="NDVI")
plt.title("NDVI from Landsat 8 (SR_B4 & SR_B5)")
plt.show()