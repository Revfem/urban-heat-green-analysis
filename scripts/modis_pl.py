import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt

# ====== EDITAȚI AICI ======
EVI_TIF   = r"data/MODIS/MOD13Q1.A2024193.h19v04.061.2024212093315_MODIS_Grid_16DAY_250m_500m_EFI.tif"                    # EVI GeoTIFF (nu HDF)
LST_REF   = r"output/MOD11A2_LST_Day_celsius.tif"            # LST 1 km (°C) - referință
OUT_EVI1K = r"output/MOD13Q1_EVI_1km_aligned.tif"            # EVI aliniat 1 km
# opțional pentru stack:
NDVI1K    = r"output/MOD13Q1_NDVI_1km_aligned.tif"           # dacă ai deja NDVI la 1 km
OUT_STACK = r"output/stack_NDVI_EVI_LST_1km.tif"
# ==========================

os.makedirs("output", exist_ok=True)

# 1) Citește EVI GeoTIFF
with rasterio.open(EVI_TIF) as src:
    evi_raw = src.read(1).astype(np.float32)
    prof_evi = src.profile
    evi_nodata = src.nodata

# 2) Curățare + auto-scalare
#   MOD13Q1 EVI raw: valori întregi cu factor 0.0001; "gata scalat" ~ [-1..1].
mask_invalid = np.zeros_like(evi_raw, dtype=bool)
if evi_nodata is not None:
    mask_invalid |= (evi_raw == evi_nodata)
# uneori "fill" <= -2000; protejăm:
mask_invalid |= (evi_raw <= -2000)
evi_raw = np.where(mask_invalid, np.nan, evi_raw)

# auto-detect scală: dacă max > 1.5, probabil e "raw" (×0.0001 necesar)
finite = np.isfinite(evi_raw)
maxv = np.nan if not finite.any() else np.nanmax(evi_raw)
if np.isnan(maxv):
    raise SystemExit("EVI are doar NaN/NoData. Verifică fișierul.")
if maxv > 1.5:
    evi_scaled = evi_raw * 0.0001
else:
    evi_scaled = evi_raw  # deja în [-1..1]

# clamp minor
evi_scaled = np.clip(evi_scaled, -1.0, 1.0)

# 3) Reproiectăm la 1 km, aliniat cu LST (folosim average la downsample)
with rasterio.open(LST_REF) as ref:
    ref_prof = ref.profile
    dst_crs = ref.crs
    dst_transform = ref.transform
    H, W = ref.height, ref.width

evi_1km = np.full((H, W), np.nan, dtype=np.float32)
reproject(
    source=evi_scaled,
    destination=evi_1km,
    src_transform=prof_evi["transform"],
    src_crs=prof_evi["crs"],
    dst_transform=dst_transform,
    dst_crs=dst_crs,
    resampling=Resampling.average,
    src_nodata=np.nan,
    dst_nodata=np.nan,
)

# 4) Salvăm EVI 1 km
out_prof = ref_prof.copy()
out_prof.update(dtype=rasterio.float32, count=1, nodata=np.nan, compress="DEFLATE")
with rasterio.open(OUT_EVI1K, "w", **out_prof) as dst:
    dst.write(evi_1km.astype(np.float32), 1)
print("Saved:", OUT_EVI1K)

# 5) (Opțional) Stack NDVI + EVI + LST (toate la 1 km)
try:
    with rasterio.open(NDVI1K) as s:
        ndvi_1km = s.read(1).astype(np.float32)
    with rasterio.open(LST_REF) as s:
        lst_1km = s.read(1).astype(np.float32)
    assert ndvi_1km.shape == evi_1km.shape == lst_1km.shape
    stack_prof = out_prof.copy()
    stack_prof.update(count=3)
    with rasterio.open(OUT_STACK, "w", **stack_prof) as dst:
        dst.write(ndvi_1km, 1)  # band1: NDVI
        dst.write(evi_1km,  2)  # band2: EVI
        dst.write(lst_1km,  3)  # band3: LST (°C)
    print("Saved stack:", OUT_STACK)
except Exception as e:
    print("Skip stack:", e)

# 6) Plot verificare
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
im0 = ax[0].imshow(evi_scaled, vmin=-1, vmax=1, cmap="RdYlGn")
ax[0].set_title("EVI original (auto-scaled)")
plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
im1 = ax[1].imshow(evi_1km, vmin=-1, vmax=1, cmap="RdYlGn")
ax[1].set_title("EVI 1 km (aligned LST)")
plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
