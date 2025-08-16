
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import argparse
import os
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def read(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        prof = src.profile
    return arr, prof

def write(path, arr, prof, nodata=np.nan):
    p = prof.copy()
    p.update(dtype='float32', count=1, compress='deflate', nodata=nodata, driver='GTiff')
    with rasterio.open(path, 'w', **p) as dst:
        dst.write(arr.astype(np.float32), 1)

def reproject_to(ref_prof, arr, prof, resampling=Resampling.bilinear):
    dst = np.empty((ref_prof['height'], ref_prof['width']), dtype=np.float32)
    reproject(arr, dst,
              src_transform=prof['transform'], src_crs=prof['crs'],
              dst_transform=ref_prof['transform'], dst_crs=ref_prof['crs'],
              dst_width=ref_prof['width'], dst_height=ref_prof['height'],
              resampling=resampling)
    return dst

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--s2_ndvi', required=True)
    ap.add_argument('--s2_ndbi', required=True)
    ap.add_argument('--s2_ndwi', required=True)
    ap.add_argument('--l8_lst', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ndvi, s2_prof = read(args.s2_ndvi)
    ndbi, _ = read(args.s2_ndbi)
    ndwi, _ = read(args.s2_ndwi)
    lst30, l8_prof = read(args.l8_lst)

    # Reproject L8 LST to S2 10m grid
    lst10 = reproject_to(s2_prof, lst30, l8_prof, Resampling.bilinear)

    # Build training matrix
    X = np.stack([ndvi, ndbi, ndwi], axis=-1)
    y = lst10

    valid = np.isfinite(X).all(axis=-1) & np.isfinite(y)
    Xv = X[valid]
    yv = y[valid]

    if Xv.size == 0:
        raise RuntimeError("No overlapping valid pixels between S2 indices and L8 LST. Check masks and AOI/time matching.")

    # Holdout split (random subset)
    rng = np.random.default_rng(42)
    idx = np.arange(Xv.shape[0])
    rng.shuffle(idx)
    n_train = int(0.8 * len(idx))
    tr, te = idx[:n_train], idx[n_train:]
    Xtr, Xte = Xv[tr], Xv[te]
    ytr, yte = yv[tr], yv[te]

    model = HuberRegressor()
    model.fit(Xtr, ytr)
    ypred = model.predict(Xte)

    r2 = r2_score(yte, ypred)
    mae = mean_absolute_error(yte, ypred)

    # Predict full map
    yhat = np.full_like(y, np.nan, dtype=np.float32)
    yhat[valid] = model.predict(Xv).astype(np.float32)

    # SUHI: anomaly vs rural baseline (NDBI<0 & NDVI>0.5)
    rural_mask = (ndbi < 0.0) & (ndvi > 0.5) & np.isfinite(yhat)
    if rural_mask.sum() >= 50:
        rural_med = np.nanmedian(yhat[rural_mask])
    else:
        rural_med = np.nanmedian(yhat[np.isfinite(yhat)])
    suhi = yhat - rural_med

    # Save
    write(os.path.join(args.out_dir, 'lst_pred_from_s2_10m.tif'), yhat, s2_prof)
    write(os.path.join(args.out_dir, 'suhi_pred_10m.tif'), suhi, s2_prof)

    # Save a small report
    with open(os.path.join(args.out_dir, 'uhi_fit_report.txt'), 'w') as f:
        f.write(f'HuberRegressor on features [NDVI, NDBI, NDWI]\n')
        f.write(f'R2={r2:.3f}, MAE={mae:.2f} °C, rural_baseline={rural_med:.2f} °C\n')
        f.write(f'Coefficients: {model.coef_.tolist()}  Intercept: {model.intercept_:.3f}\n')

    print(f'Done. R2={r2:.3f}, MAE={mae:.2f} °C. Rural baseline {rural_med:.2f} °C.')

if __name__ == '__main__':
    main()
