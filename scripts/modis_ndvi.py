import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt


NDVI_TIF = r"data/MODIS/MOD13Q1.A2024193.h19v04.061.2024212093315_MODIS_Grid_16DAY_250m_500m_VI.tif"  
LST_REF  = r"output/MOD11A2_LST_Day_celsius.tif"                    
OUT_1KM  = r"output/MOD13Q1_NDVI_1km_aligned.tif"                   
OUT_STACK = r"output/stack_LSTday_NDVI_1km.tif"                     

os.makedirs("output", exist_ok=True)


with rasterio.open(NDVI_TIF) as src_ndvi:
    ndvi_raw = src_ndvi.read(1).astype(np.float32)
    prof_ndvi = src_ndvi.profile
    ndvi_nodata = src_ndvi.nodata

# MOD13Q1: de obicei NDVI are nodata ~ -3000 / 0; orice <= -2000 considerăm invalid
mask_invalid = np.zeros_like(ndvi_raw, dtype=bool)
if ndvi_nodata is not None:
    mask_invalid |= (ndvi_raw == ndvi_nodata)
mask_invalid |= (ndvi_raw <= -2000)

ndvi_raw = np.where(mask_invalid, np.nan, ndvi_raw)

# Scale la [-1..1] aproximativ
ndvi_scaled = ndvi_raw * 0.0001
# clamp ușor ca să eviți outliers rari
ndvi_scaled = np.clip(ndvi_scaled, -1.0, 1.0)

# === 2) Aducem NDVI la grila LST (1km) cu Resampling.average ===
with rasterio.open(LST_REF) as ref:
    ref_prof = ref.profile
    H, W = ref.height, ref.width
    dst_transform = ref.transform
    dst_crs = ref.crs

ndvi_1km = np.full((H, W), np.nan, dtype=np.float32)

reproject(
    source=ndvi_scaled,
    destination=ndvi_1km,
    src_transform=prof_ndvi["transform"],
    src_crs=prof_ndvi["crs"],
    dst_transform=dst_transform,
    dst_crs=dst_crs,
    resampling=Resampling.average,  
    src_nodata=np.nan,
    dst_nodata=np.nan,
)

# === 3) Salvăm NDVI 1km aliniat cu LST ===
out_prof = ref_prof.copy()
out_prof.update(dtype=rasterio.float32, count=1, nodata=np.nan, compress="DEFLATE")

with rasterio.open(OUT_1KM, "w", **out_prof) as dst:
    dst.write(ndvi_1km.astype(np.float32), 1)

print("Saved:", OUT_1KM)

# === 4) (Opțional) Stack NDVI + LST Day într-un singur fișier ===
try:
    with rasterio.open(LST_REF) as s:
        lst_day = s.read(1).astype(np.float32)  # deja în °C
        # asigurare dimensiuni identice
        assert lst_day.shape == ndvi_1km.shape
    stack_prof = out_prof.copy()
    stack_prof.update(count=2)
    with rasterio.open(OUT_STACK, "w", **stack_prof) as dst:
        dst.write(ndvi_1km.astype(np.float32), 1)  # band 1: NDVI
        dst.write(lst_day.astype(np.float32), 2)   # band 2: LST Day (°C)
    print("Saved stack:", OUT_STACK)
except Exception as e:
    print("Skip stack (could not open LST_REF or mismatch):", e)

# === 5) Plot rapid ===
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
im0 = ax[0].imshow(ndvi_scaled, vmin=-1, vmax=1, cmap="RdYlGn")
ax[0].set_title("NDVI (original scale×0.0001)")
plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

im1 = ax[1].imshow(ndvi_1km, vmin=-1, vmax=1, cmap="RdYlGn")
ax[1].set_title("NDVI resamplat la 1 km (aligned LST)")
plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
