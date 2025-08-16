import os, numpy as np, rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from rasterio import band
from sklearn.linear_model import LinearRegression
import argparse

def read1(path):
    with rasterio.open(path) as src:
        prof = src.profile
    return prof

def reproject_tile_to_s2(lst_ds, dst_h, dst_w, dst_transform, dst_crs):
    """Reproiectează o bucată L8 LST în fereastra S2 (bilinear)."""
    dst = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
    reproject(
        source=band(lst_ds, 1),
        destination=dst,
        src_transform=lst_ds.transform,
        src_crs=lst_ds.crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_width=dst_w,
        dst_height=dst_h,
        resampling=Resampling.bilinear,
        src_nodata=lst_ds.nodata,
        dst_nodata=np.nan,
    )
    return dst

def normalize(arr):
    mn = np.nanmin(arr); mx = np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn + 1e-6)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2_ndvi", required=True)
    ap.add_argument("--s2_ndbi", required=True)
    ap.add_argument("--l8_lst", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tile", type=int, default=512)          # micșorează dacă e nevoie (256)
    ap.add_argument("--sample_max", type=int, default=120_000) # scade dacă vrei mai rapid
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Deschidem S2 indices + L8 LST
    ndvi_ds = rasterio.open(args.s2_ndvi)
    ndbi_ds = rasterio.open(args.s2_ndbi)
    lst_ds  = rasterio.open(args.l8_lst)

    assert (ndvi_ds.crs == ndbi_ds.crs) and (ndvi_ds.transform == ndbi_ds.transform), "NDVI și NDBI trebuie pe același grid."
    H, W = ndvi_ds.height, ndvi_ds.width
    s2_transform, s2_crs = ndvi_ds.transform, ndvi_ds.crs

    # === 1) Colectăm eșantioane pentru model (tile-wise, cu plafon)
    X_list, y_list = [], []
    rng = np.random.default_rng(42)
    TILE = args.tile
    SAMPLE_MAX = args.sample_max

    total = 0
    for r0 in range(0, H, TILE):
        h = min(TILE, H - r0)
        for c0 in range(0, W, TILE):
            w = min(TILE, W - c0)
            win = Window(c0, r0, w, h)
            # citim indici S2
            ndvi_blk = ndvi_ds.read(1, window=win).astype(np.float32)
            ndbi_blk = ndbi_ds.read(1, window=win).astype(np.float32)
            # reproiectăm L8 LST pe fereastra S2
            dst_transform = rasterio.windows.transform(win, s2_transform)
            lst_blk = reproject_tile_to_s2(lst_ds, h, w, dst_transform, s2_crs)

            mask = np.isfinite(ndvi_blk) & np.isfinite(ndbi_blk) & np.isfinite(lst_blk)
            if not mask.any(): 
                continue

            x1 = ndvi_blk[mask].ravel()
            x2 = ndbi_blk[mask].ravel()
            y  = lst_blk[mask].ravel()

            n = x1.size
            # ținem eșantion mic per fereastră
            keep = min(3000, n)
            idx = rng.choice(n, size=keep, replace=False)
            X_list.append(np.column_stack([x1[idx], x2[idx]]))
            y_list.append(y[idx])

            total += keep
            if total >= SAMPLE_MAX:
                break
        if total >= SAMPLE_MAX:
            break

    if not X_list:
        raise RuntimeError("Nu s-au găsit pixeli valizi comuni între S2 și L8.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # === 2) Antrenăm model simplu (rapid)
    model = LinearRegression()
    model.fit(X, y)
    print("Model: LST = a*NDVI + b*NDBI + c")
    print(" a =", float(model.coef_[0]), " b =", float(model.coef_[1]), " c =", float(model.intercept_))

    # baseline rural din eșantioane (NDBI<0 & NDVI>0.5)
    rural_mask = (X[:,1] < 0) & (X[:,0] > 0.5)
    if rural_mask.any():
        rural_baseline = np.nanmedian(model.predict(X[rural_mask]))
    else:
        rural_baseline = np.nanmedian(y)
    print("Rural baseline (°C):", float(rural_baseline))

    # === 3) Predicție și scriere tile-wise
    out_prof = ndvi_ds.profile.copy()
    out_prof.update(dtype=rasterio.float32, count=1, nodata=np.nan, compress="DEFLATE")
    out_lst_path = os.path.join(args.out_dir, "lst_pred_from_s2_10m.tif")
    out_suhi_path = os.path.join(args.out_dir, "suhi_pred_10m.tif")

    with rasterio.open(out_lst_path, "w", **out_prof) as out_lst, \
         rasterio.open(out_suhi_path, "w", **out_prof) as out_suhi:

        for r0 in range(0, H, TILE):
            h = min(TILE, H - r0)
            for c0 in range(0, W, TILE):
                w = min(TILE, W - c0)
                win = Window(c0, r0, w, h)
                ndvi_blk = ndvi_ds.read(1, window=win).astype(np.float32)
                ndbi_blk = ndbi_ds.read(1, window=win).astype(np.float32)

                X_blk = np.column_stack([ndvi_blk.ravel(), ndbi_blk.ravel()])
                valid = np.isfinite(X_blk).all(axis=1)

                pred = np.full(X_blk.shape[0], np.nan, dtype=np.float32)
                if valid.any():
                    pred[valid] = model.predict(X_blk[valid]).astype(np.float32)
                pred = pred.reshape(h, w)

                suhi_blk = pred - rural_baseline
                out_lst.write(pred.astype(np.float32), 1, window=win)
                out_suhi.write(suhi_blk.astype(np.float32), 1, window=win)

    ndvi_ds.close(); ndbi_ds.close(); lst_ds.close()
    print("Saved:", out_lst_path)
    print("Saved:", out_suhi_path)

if __name__ == "__main__":
    main()
