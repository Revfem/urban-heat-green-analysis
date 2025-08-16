import os
import gdown
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling
import shutil

# === 0) Curățenie ===
shutil.rmtree("data", ignore_errors=True)

# === 1) Link-uri fișiere ===
# Sentinel-2 L2A B04 (10 m) - referință grilă (poți pune și B08 dacă preferi)
url_s2_b04 = "https://drive.google.com/file/d/1wOKuPRr-srXuqVMalo1KJuFD3FAdiBHV/view?usp=drive_link"       
# (opțional) Sentinel-2 QA60 (60 m) pt. nori (bit 10/11)
url_s2_qa60 = " "     # poți lăsa gol dacă nu vrei mască

# MODIS LST (MOD11A2 zi) ca GeoTIFF sau subset deja extras (1 km)
url_modis_lst = "PUT_MODIS_LST_GEOTIFF_LINK_AICI"

# === 2) Căi locale ===
s2_b04_path = "data/S2/B04_10m.jp2"
s2_qa60_path = "data/S2/QA60_60m.jp2"
modis_lst_path = "data/MODIS/LST_day_1km.tif"

os.makedirs(os.path.dirname(s2_b04_path), exist_ok=True)
os.makedirs(os.path.dirname(modis_lst_path), exist_ok=True)

def dl(url, outp):
    if url and not os.path.exists(outp):
        gdown.download(url=url, output=outp, quiet=False, fuzzy=True)

dl(url_s2_b04, s2_b04_path)
if url_s2_qa60:
    dl(url_s2_qa60, s2_qa60_path)
dl(url_modis_lst, modis_lst_path)

# === 3) Citim grila de referință din S2 B04 (10 m) ===
with rasterio.open(s2_b04_path) as ref:
    ref_profile = ref.profile
    ref_crs = ref_profile["crs"]
    ref_transform = ref_profile["transform"]
    ref_height, ref_width = ref_profile["height"], ref_profile["width"]
    # doar pentru vizual: citim și B04
    b04 = ref.read(1).astype(np.float32)
    # normalizăm la [0,1] pentru un plot mai prietenos
    if b04.max() > 1.0:
        b04 = b04 / 10000.0

# === 4) Citim MODIS LST, convertim la °C și reprojectăm pe grila S2 ===
with rasterio.open(modis_lst_path) as src:
    lst_raw = src.read(1).astype(np.float32)
    src_profile = src.profile
    src_nodata = src_profile.get("nodata", None)

# MOD11A2 scale: 0.02 K; 0 = fill; valori > 0 valide
# convertim la Kelvin și apoi la °C; punem NaN unde nu e valid
lst_raw = np.where(lst_raw <= 0, np.nan, lst_raw)
lst_kelvin = lst_raw * 0.02
lst_c = lst_kelvin - 273.15

# reproject pe grila S2 (10 m)
lst_on_s2 = np.full((ref_height, ref_width), np.nan, dtype=np.float32)
reproject(
    source=lst_c,
    destination=lst_on_s2,
    src_transform=src_profile["transform"],
    src_crs=src_profile["crs"],
    dst_transform=ref_transform,
    dst_crs=ref_crs,
    resampling=Resampling.bilinear
)

# === 5) (opțional) mască de nori din QA60 ===
# QA60: bit 10 = clouds, bit 11 = cirrus
if url_s2_qa60 and os.path.exists(s2_qa60_path):
    with rasterio.open(s2_qa60_path) as qa:
        qa60 = qa.read(1)
        qa_profile = qa.profile

    # reproiectăm QA60 la 10 m (nearest, ca e mască binară)
    qa_on_s2 = np.zeros((ref_height, ref_width), dtype=np.uint16)
    reproject(
        source=qa60,
        destination=qa_on_s2,
        src_transform=qa_profile["transform"],
        src_crs=qa_profile["crs"],
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.nearest
    )
    clouds = ((qa_on_s2 & (1 << 10)) != 0) | ((qa_on_s2 & (1 << 11)) != 0)
    lst_on_s2[clouds] = np.nan

# === 6) Salvăm LST pe grila S2 ===
os.makedirs("output", exist_ok=True)
out_path = "output/temperature_celsius_s2grid.tif"
out_prof = ref_profile.copy()
out_prof.update(dtype=rasterio.float32, count=1, compress="DEFLATE", nodata=np.nan)

with rasterio.open(out_path, "w", **out_prof) as dst:
    dst.write(lst_on_s2.astype(np.float32), 1)

print("Saved:", out_path)

# === 7) Plot rapid ===
plt.figure(figsize=(12, 4))
ax1 = plt.subplot(1, 2, 1)
im1 = ax1.imshow(b04, cmap="gray")
ax1.set_title("Sentinel-2 B04 (10 m)")
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

ax2 = plt.subplot(1, 2, 2)
# setăm o scară rezonabilă de vară, ajustează după nevoie
vmin, vmax = 20, 45
im2 = ax2.imshow(lst_on_s2, cmap="inferno", vmin=vmin, vmax=vmax)
ax2.set_title("Temperatură suprafață (°C) pe grila S2")
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
