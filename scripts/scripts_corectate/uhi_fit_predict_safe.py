import argparse, os, numpy as np, rasterio
from rasterio.windows import Window
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from joblib import dump

def read1(path): 
    ds = rasterio.open(path); return ds

def _is_valid(arr, nodata):
    m = np.isfinite(arr)
    if nodata is not None:
        m &= (arr != nodata)
    return m

def valid_mask_feats(ndvi, ndvi_nod, ndbi, ndbi_nod, ndwi, ndwi_nod):
    return _is_valid(ndvi, ndvi_nod) & _is_valid(ndbi, ndbi_nod) & _is_valid(ndwi, ndwi_nod)

def sample_blocked(dss, sample_max=1_500_000, per_block=20000, rng=42):
    """
    dss: dict cu rasterio datasets: 'ndvi','ndbi','ndwi','lst'
    Maschează: feature NODATA + LST NODATA în y.
    """
    rs = np.random.RandomState(rng)
    Xs, ys = [], []
    tmpl = dss["ndvi"]

    ndvi_nod = dss["ndvi"].nodata
    ndbi_nod = dss["ndbi"].nodata
    ndwi_nod = dss["ndwi"].nodata
    lst_nod  = dss["lst"].nodata

    for _, w in tmpl.block_windows(1):
        ndvi = dss["ndvi"].read(1, window=w).astype("float32")
        ndbi = dss["ndbi"].read(1, window=w).astype("float32")
        ndwi = dss["ndwi"].read(1, window=w).astype("float32")
        lst  = dss["lst"].read(1, window=w).astype("float32")

        m_feat = valid_mask_feats(ndvi, ndvi_nod, ndbi, ndbi_nod, ndwi, ndwi_nod)
        m_y    = _is_valid(lst, lst_nod) & np.isfinite(lst) & (lst > -50) & (lst < 80)  # gardă de siguranță
        m      = m_feat & m_y
        if not m.any():
            continue

        idx = np.flatnonzero(m)
        if idx.size > per_block:
            idx = rs.choice(idx, size=per_block, replace=False)

        Xs.append(np.stack([ndvi.flat[idx], ndbi.flat[idx], ndwi.flat[idx]], axis=1))
        ys.append(lst.flat[idx])

        if sum(x.shape[0] for x in Xs) >= sample_max:
            break

    if not Xs:
        raise SystemExit("Niciun pixel valid pentru training (verifică NODATA și overlap-ul).")
    X = np.concatenate(Xs, axis=0)[:sample_max].astype("float32")
    y = np.concatenate(ys, axis=0)[:sample_max].astype("float32")
    return X, y

def predict_tilewise(dss, model, out_path):
    tmpl = dss["ndvi"]
    prof = tmpl.profile.copy()
    prof.update(driver="GTiff", dtype="float32", count=1, nodata=-9999,
                tiled=True, blockxsize=512, blockysize=512, compress="DEFLATE", BIGTIFF="IF_SAFER")

    ndvi_nod = dss["ndvi"].nodata
    ndbi_nod = dss["ndbi"].nodata
    ndwi_nod = dss["ndwi"].nodata

    with rasterio.open(out_path, "w", **prof) as dst:
        for _, w in tmpl.block_windows(1):
            ndvi = dss["ndvi"].read(1, window=w).astype("float32")
            ndbi = dss["ndbi"].read(1, window=w).astype("float32")
            ndwi = dss["ndwi"].read(1, window=w).astype("float32")

            m = valid_mask_feats(ndvi, ndvi_nod, ndbi, ndbi_nod, ndwi, ndwi_nod)
            pred = np.full(ndvi.size, np.nan, dtype="float32")
            if m.any():
                X = np.stack([ndvi[m], ndbi[m], ndwi[m]], axis=1).astype("float32")
                pred[m.ravel()] = model.predict(X).astype("float32")

            pred = pred.reshape(ndvi.shape)
            pred = np.where(np.isfinite(pred), pred, -9999).astype("float32")
            dst.write(pred, 1, window=w)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2_ndvi", required=True)
    ap.add_argument("--s2_ndbi", required=True)
    ap.add_argument("--s2_ndwi", required=True)
    ap.add_argument("--lst",     required=True)  # LST deja pe grila S2 (10 m) – MODIS/S3/L8 resamplat
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sample_max", type=int, default=1_500_000)
    ap.add_argument("--per_block",  type=int, default=20000)
    ap.add_argument("--alpha", type=float, default=1.0)  # Ridge
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dss = {
        "ndvi": read1(args.s2_ndvi),
        "ndbi": read1(args.s2_ndbi),
        "ndwi": read1(args.s2_ndwi),
        "lst":  read1(args.lst),
    }
    # sanity: grile identice
    W,H = dss["ndvi"].width, dss["ndvi"].height
    for k,v in dss.items():
        assert v.width==W and v.height==H and v.transform==dss["ndvi"].transform and v.crs==dss["ndvi"].crs, f"Grid mismatch at {k}"

    # === SAMPLE SAFE ===
    X, y = sample_blocked(dss, sample_max=args.sample_max, per_block=args.per_block)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)

    # model ușor și stabil
    mdl = Ridge(alpha=args.alpha)  # float32 intră ok, intern va folosi float64 dar pe lot mic
    mdl.fit(Xtr, ytr)
    r2 = mdl.score(Xva, yva)
    mae = float(np.mean(np.abs(mdl.predict(Xva) - yva)))
    print(f"[Ridge] R2={r2:.3f}  MAE={mae:.3f} °C  (n_train={len(ytr):,}, n_val={len(yva):,})")

    # salvează modelul
    dump(mdl, os.path.join(args.out_dir, "uhi_model.joblib"))

    # === PREDICT TILE-WISE ===
    out_pred = os.path.join(args.out_dir, "lst_pred_10m.tif")
    predict_tilewise(dss, mdl, out_pred)

    # cleanup
    for v in dss.values(): v.close()

if __name__ == "__main__":
    main()
