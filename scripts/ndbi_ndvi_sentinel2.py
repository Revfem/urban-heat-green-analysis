import os
import shutil
import gdown
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling


shutil.rmtree("data", ignore_errors=True)


url_b4 = "https://drive.google.com/file/d/1wOKuPRr-srXuqVMalo1KJuFD3FAdiBHV/view?usp=drive_link"  
url_b8 = "https://drive.google.com/file/d/1-OhBvYDZP6x5Dq2Ram7Tx9xf5hrlugIq/view?usp=drive_link"   
url_b11 = "https://drive.google.com/file/d/1wKq5eG4Fggy4wvFKSWTJlIVHx9zLp14l/view?usp=drive_link" 


b4_path = "data/S2/B04_10m.jp2"
b8_path = "data/S2/B08_10m.jp2"
b11_path = "data/S2/B11_20m.jp2"
os.makedirs(os.path.dirname(b4_path), exist_ok=True)


def download(url, out_path):
    if not os.path.exists(out_path):
        gdown.download(url=url, output=out_path, quiet=False, fuzzy=True)
    return out_path

download(url_b4, b4_path)
download(url_b8, b8_path)
download(url_b11, b11_path)

with rasterio.open(b4_path) as src:
    red = src.read(1).astype(np.float32) / 10000.0
    profile_10m = src.profile

gtiff_profile = {
    "driver": "GTiff",
    "height": profile_10m["height"],
    "width":  profile_10m["width"],
    "count":  1,
    "dtype":  rasterio.float32,
    "crs":    profile_10m["crs"],
    "transform": profile_10m["transform"],
    "compress": "DEFLATE"
}

with rasterio.open(b8_path) as src:
    nir = src.read(1).astype(np.float32) / 10000.0


with rasterio.open(b11_path) as src:
    swir_10m = np.empty((profile_10m["height"], profile_10m["width"]), dtype=np.float32)
    reproject(
        source=src.read(1).astype(np.float32) / 10000.0,
        destination=swir_10m,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=profile_10m["transform"],
        dst_crs=profile_10m["crs"],
        resampling=Resampling.bilinear
    )


ndvi = (nir - red) / (nir + red + 1e-6)
ndbi = (swir_10m - nir) / (swir_10m + nir + 1e-6)


profile_10m.update(dtype=rasterio.float32)
os.makedirs("output", exist_ok=True)

with rasterio.open("output/ndvi_s2.tif", "w", **gtiff_profile ) as dst:
    dst.write(ndvi.astype(np.float32), 1)

with rasterio.open("output/ndbi_s2.tif", "w", **gtiff_profile ) as dst:
    dst.write(ndbi.astype(np.float32), 1)

print("[✓] NDVI și NDBI salvate în folderul output/")

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# im1 = axes[0].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
# axes[0].set_title("NDVI (S2)")
# axes[0].axis("off")
# fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="NDVI")

# im2 = axes[1].imshow(ndbi, cmap="BrBG", vmin=-1, vmax=1)
# axes[1].set_title("NDBI (S2)")
# axes[1].axis("off")
# fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="NDBI")

# plt.tight_layout()
# plt.show()