import os
import shutil
import gdown
import numpy as np
import rasterio
import matplotlib.pyplot as plt


file_url_b4 = "https://drive.google.com/file/d/1-ZuaGxkiRKRi-vnSDGqVFUVpwBT7LkMk/view?usp=drive_link"  # Red (SR_B4)
file_url_b5 = "https://drive.google.com/file/d/1DF6o4o0ifd-kwfJr4EYZBxpiaVEglbXD/view?usp=drive_link"  # NIR (SR_B5)

shutil.rmtree("data/LANDSAT8", ignore_errors=True)


b4_path = "data/LANDSAT8/LC08_L2SP_182029_20240811/LC08_L2SP_182029_20240815_02_T1_SR_B4.TIF"
b5_path = "data/LANDSAT8/LC08_L2SP_182029_20240811/LC08_L2SP_182029_20240815_02_T1_SR_B5.TIF"
os.makedirs(os.path.dirname(b4_path), exist_ok=True)

def download_if_missing(url, out_path):
    if not os.path.exists(out_path):
        gdown.download(url=url, output=out_path, quiet=False, fuzzy=True)
    return out_path

download_if_missing(file_url_b4, b4_path)
download_if_missing(file_url_b5, b5_path)


scale = 0.0000275
offset = -0.2

with rasterio.open(b5_path) as s_nir:
    nir = s_nir.read(1).astype(np.float32)
    profile = s_nir.profile

with rasterio.open(b4_path) as s_red:
    red = s_red.read(1).astype(np.float32)

nir = nir * scale + offset
red = red * scale + offset

nir = np.clip(nir, 0, 1)
red = np.clip(red, 0, 1)

ndvi = (nir - red) / (nir + red + 1e-6)

profile.update(dtype=rasterio.float32)
os.makedirs("output", exist_ok=True)
out_path = "output/ndvi_l8.tif"

with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(ndvi.astype(np.float32), 1)

# Plot (opțional)
plt.figure(figsize=(8, 6))
im = plt.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
plt.colorbar(im, label="NDVI")
plt.title("NDVI from Landsat 8 (SR_B4 & SR_B5)")
plt.tight_layout()
plt.show()

print("Saved:", out_path)
