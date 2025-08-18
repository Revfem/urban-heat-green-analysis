import argparse, numpy as np, rasterio

p = argparse.ArgumentParser()
p.add_argument("--lst", required=True)             # LST (predicție) pe 10 m
p.add_argument("--ndvi", required=True)            # NDVI pe 10 m (pt. rural)
p.add_argument("--built_mask", required=False)     # opțional: 1=urban (ex. built_mask_10m.tif)
p.add_argument("--water_mask", required=False)     # opțional: 1=apă
p.add_argument("--ndvi_thr", type=float, default=0.5)   # threshold pt. rural
p.add_argument("--out", required=True)
args = p.parse_args()

def read1(path):
    with rasterio.open(path) as ds:
        a = ds.read(1).astype("float32"); nod = ds.nodata; prof = ds.profile
    if nod is not None: a[a==nod] = np.nan
    return a, prof

lst, prof = read1(args.lst)
ndvi,_    = read1(args.ndvi)

mask_rural = ndvi > args.ndvi_thr
if args.built_mask:
    with rasterio.open(args.built_mask) as ds: built = ds.read(1) == 1
    mask_rural &= ~built
if args.water_mask:
    with rasterio.open(args.water_mask) as ds: water = ds.read(1) == 1
    mask_rural &= ~water

if not np.any(mask_rural):
    raise SystemExit("Nu există pixeli 'rural' după filtre. Scade --ndvi_thr sau renunță la built/water mask.")

baseline = float(np.nanmedian(lst[mask_rural]))
suhi = lst - baseline

prof.update(dtype="float32", count=1, nodata=-9999)
with rasterio.open(args.out, "w", **prof) as dst:
    dst.write(np.where(np.isfinite(suhi), suhi, -9999).astype("float32"), 1)
print(f"Baseline rural = {baseline:.2f} °C  -> {args.out}")
