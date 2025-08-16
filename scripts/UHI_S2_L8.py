# UHI_S2_L8_tilewise.py  (RAM-safe, no plots)
import os
import math
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from rasterio import band
from sklearn.linear_model import LinearRegression  # poți schimba pe Ridge/RandomForest

# ---- INPUTS
NDVI_S2 = "output/ndvi_s2.tif"
NDBI_S2 = "output/ndbi_s2.tif"
LST_L8  = "output/temperature_celsius.tif"  # °C, din scriptul Landsat

# ---- OUTPUTS
OUT_LST10 = "output/lst_pred_10m_from_s2.tif"
OUT_UHMI  = "output/uhmi_10m.tif"

# ---- PERFORMANCE KNOBS (ajustează după PC)
TILE = 1024          # mărimea ferestrei (px). 512 sau 2048 merg ok.
SAMPLE_MAX = 200_000 # nr. maxim de pixeli folosiți la antrenare (limitează RAM/CPU)

def normalize(arr):
    mn = np.nanmin(arr); mx = np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn + 1e-6)

os.makedirs("output", exist_ok=True)

# ---- Deschidem sursele
with rasterio.open(NDVI_S2) as ndvi_ds, \
     rasterio.open(NDBI_S2) as ndbi_ds, \
     rasterio.open(LST_L8) as lst_ds:

    H, W = ndvi_ds.height, ndvi_ds.width
    transform_s2 = ndvi_ds.transform
    crs_s2 = ndvi_ds.crs

    # verificări rapide
    assert (ndvi_ds.crs == ndbi_ds.crs) and (ndvi_ds.transform == ndbi_ds.transform) \
        and (ndvi_ds.height == ndbi_ds.height) and (ndvi_ds.width == ndbi_ds.width), \
        "NDVI și NDBI trebuie să fie pe același grid (rulează scriptul S2 corect)."

    # ---- 1) STRÂNGEM EȘANTIOANE pentru model, pe ferestre
    X_list = []
    y_list = []
    rng = np.random.default_rng(42)

    # pregătim o „bandă” sursă LST pentru reproject on-the-fly
    lst_band = band(lst_ds, 1)

    for row0 in range(0, H, TILE):
        h = min(TILE, H - row0)
        for col0 in range(0, W, TILE):
            w = min(TILE, W - col0)
            win = Window(col_off=col0, row_off=row0, width=w, height=h)

            # citim NDVI/NDBI pe fereastră
            ndvi_blk = ndvi_ds.read(1, window=win).astype(np.float32)
            ndbi_blk = ndbi_ds.read(1, window=win).astype(np.float32)

            # reproiectăm LST în fereastra S2 (bilinear)
            lst_blk = np.full((h, w), np.nan, dtype=np.float32)
            # transform pentru fereastra curentă
            dst_transform = rasterio.windows.transform(win, transform_s2)

            reproject(
                source=lst_band,
                destination=lst_blk,
                src_transform=lst_ds.transform,
                src_crs=lst_ds.crs,
                dst_transform=dst_transform,
                dst_crs=crs_s2,
                resampling=Resampling.bilinear,
                src_nodata=lst_ds.nodata,
                dst_nodata=np.nan,
            )

            # mască pixeli valizi
            mask = (~np.isnan(ndvi_blk)) & (~np.isnan(ndbi_blk)) & (~np.isnan(lst_blk)) \
                   & np.isfinite(ndvi_blk) & np.isfinite(ndbi_blk) & np.isfinite(lst_blk)
            if not mask.any():
                continue

            # extragem eșantioane
            x1 = ndvi_blk[mask].ravel()
            x2 = ndbi_blk[mask].ravel()
            y  = lst_blk[mask].ravel()

            # downsample aleator dacă sunt prea multe în fereastra curentă
            n = x1.size
            if n > 5000:  # limită per fereastră ca să nu explodăm
                idx = rng.choice(n, size=5000, replace=False)
                x1 = x1[idx]; x2 = x2[idx]; y = y[idx]

            X_list.append(np.column_stack([x1, x2]))
            y_list.append(y)

            # limită globală pe eșantioane
            total = sum(len(a) for a in y_list)
            if total >= SAMPLE_MAX:
                break
        if sum(len(a) for a in y_list) >= SAMPLE_MAX:
            break

    if not X_list:
        raise RuntimeError("Nu s-au găsit eșantioane valide pentru antrenare.")

    X_train = np.concatenate(X_list, axis=0)
    y_train = np.concatenate(y_list, axis=0)

    # dacă depășește SAMPLE_MAX, tăiem aleator
    if X_train.shape[0] > SAMPLE_MAX:
        idx = rng.choice(X_train.shape[0], size=SAMPLE_MAX, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    # ---- 2) Model simplu (poți schimba pe Ridge/RandomForest)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model: LST = a*NDVI + b*NDBI + c")
    print(" a =", float(model.coef_[0]), " b =", float(model.coef_[1]), " c =", float(model.intercept_))

    # ---- 3) Pregătim fișierele de ieșire
    out_prof = ndvi_ds.profile.copy()
    out_prof.update(count=1, dtype=rasterio.float32, nodata=np.nan, compress="DEFLATE")

    # creăm fișierele goală pentru scriere pe ferestre
    with rasterio.open(OUT_LST10, "w", **out_prof) as out_lst10, \
         rasterio.open(OUT_UHMI,  "w", **out_prof) as out_uhmi:

        # parcurgem din nou pe ferestre, prezicem și scriem blocuri
        for row0 in range(0, H, TILE):
            h = min(TILE, H - row0)
            for col0 in range(0, W, TILE):
                w = min(TILE, W - col0)
                win = Window(col_off=col0, row_off=row0, width=w, height=h)
                dst_transform = rasterio.windows.transform(win, transform_s2)

                ndvi_blk = ndvi_ds.read(1, window=win).astype(np.float32)
                ndbi_blk = ndbi_ds.read(1, window=win).astype(np.float32)

                # LST reproiectat pentru această fereastră (doar pentru normalizare UHMI dacă vrei, dar îl prezicem)
                # aici nu mai facem reproject; prezicem direct
                X_blk = np.column_stack([ndvi_blk.ravel(), ndbi_blk.ravel()])
                valid = ~np.isnan(X_blk).any(axis=1) & np.isfinite(X_blk).all(axis=1)

                lst_pred = np.full(X_blk.shape[0], np.nan, dtype=np.float32)
                if valid.any():
                    lst_pred[valid] = model.predict(X_blk[valid]).astype(np.float32)
                lst_pred = lst_pred.reshape(h, w)

                # UHMI = norm(LST_pred) * (1 - norm(NDVI))
                # normalizăm local în fereastră (robust și RAM-friendly)
                uhmi_blk = normalize(lst_pred) * (1 - normalize(ndvi_blk))

                # scriem blocurile
                out_lst10.write(lst_pred.astype(np.float32), 1, window=win)
                out_uhmi.write(uhmi_blk.astype(np.float32), 1, window=win)

print("Saved:", OUT_LST10)
print("Saved:", OUT_UHMI)
# === 4) Ploturi rapide (downsample, safe pentru RAM)
import matplotlib.pyplot as plt

# LST predicted (°C)
with rasterio.open(OUT_LST10) as src:
    lst_preview = src.read(
        out_shape=(1, src.height // 20, src.width // 20),  # reducere ~5%
        resampling=Resampling.average
    )[0]

plt.figure(figsize=(8, 6))
im1 = plt.imshow(lst_preview, cmap="inferno")
plt.title("Predicted LST (10 m, downsampled preview)")
plt.colorbar(im1, label="°C")
plt.axis("off")
plt.tight_layout()
plt.show()

# UHMI index
with rasterio.open(OUT_UHMI) as src:
    uhmi_preview = src.read(
        out_shape=(1, src.height // 20, src.width // 20),
        resampling=Resampling.average
    )[0]

plt.figure(figsize=(8, 6))
im2 = plt.imshow(uhmi_preview, cmap="magma")
plt.title("UHMI (10 m, downsampled preview)")
plt.colorbar(im2, label="UHMI")
plt.axis("off")
plt.tight_layout()
plt.show()
