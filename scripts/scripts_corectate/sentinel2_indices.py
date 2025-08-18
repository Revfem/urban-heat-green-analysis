# sentinel2_indices.py  — versiune drop-in
import argparse, os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# SCL classes
WATER_CLASS = 6
CLOUD_CLASSES = {8, 9, 10}  # medium/high + cirrus
SHADOW_CLASS = {3}
SNOW_CLASS = {11}

def read_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        prof = src.profile
    return arr, prof

def resample_to(ref_profile, src_arr, src_profile, resampling):
    dst = np.full((ref_profile["height"], ref_profile["width"]), np.nan, dtype="float32")
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_profile["transform"], src_crs=src_profile["crs"],
        dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
        resampling=resampling,
    )
    return dst

def write_tif(path, arr, profile, nodata=-9999):
    prof = profile.copy()
    prof.update(
        driver="GTiff", dtype="float32", count=1,
        nodata=nodata, compress="DEFLATE",
        tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER"
    )
    with rasterio.open(path, "w", **prof) as dst:
        out = np.where(np.isfinite(arr), arr, nodata).astype("float32")
        dst.write(out, 1)

def safe_index(a, b):
    num = (a - b)
    den = (a + b)
    out = np.full_like(a, np.nan, dtype="float32")
    ok = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[ok] = num[ok] / den[ok]
    return out

def s2_valid_and_water(scl):
    cloud  = np.isin(scl, list(CLOUD_CLASSES))
    shadow = np.isin(scl, list(SHADOW_CLASS))
    snow   = np.isin(scl, list(SNOW_CLASS))
    valid  = ~(cloud | shadow | snow)
    water  = (scl == WATER_CLASS)
    return valid, water

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b04", required=True)      # RED  (10 m)
    ap.add_argument("--b08", required=True)      # NIR  (10 m)
    ap.add_argument("--b03", required=True)      # GREEN(10 m)
    ap.add_argument("--b11", required=True)      # SWIR1(20 m)
    ap.add_argument("--scl", required=True)      # SCL  (20 m)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 10 m reference grid (RED)
    red, prof10 = read_band(args.b04)
    nir, _      = read_band(args.b08)
    green, _    = read_band(args.b03)
    # scale to reflectance
    red  /= 10000.0
    nir  /= 10000.0
    green/= 10000.0

    # 20 m → 10 m
    swir20, prof20 = read_band(args.b11)
    scl20,  sclprof= read_band(args.scl)
    swir = resample_to(prof10, swir20, prof20, Resampling.bilinear) / 10000.0
    scl  = resample_to(prof10, scl20,  sclprof, Resampling.nearest).astype("uint8")

    # masks
    valid, water = s2_valid_and_water(scl)
    mask_nd   = valid & ~water   # pentru NDVI/NDBI (exclud apa)
    mask_ndwi = valid            # pentru NDWI/MNDWI (păstrează apa)

    # indices
    ndvi  = safe_index(nir, red)     # (NIR-RED)/(NIR+RED)
    ndbi  = safe_index(swir, nir)    # (SWIR-NIR)/(SWIR+NIR)
    ndwi  = safe_index(green, nir)   # McFeeters
    mndwi = safe_index(green, swir)  # Xu

    # apply masks
    ndvi[~mask_nd]   = np.nan
    ndbi[~mask_nd]   = np.nan
    ndwi[~mask_ndwi] = np.nan
    mndwi[~mask_ndwi]= np.nan

    # write
    write_tif(os.path.join(args.out_dir, "ndvi_s2_10m.tif"),  ndvi,  prof10)
    write_tif(os.path.join(args.out_dir, "ndbi_s2_10m.tif"),  ndbi,  prof10)
    write_tif(os.path.join(args.out_dir, "ndwi_s2_10m.tif"),  ndwi,  prof10)
    write_tif(os.path.join(args.out_dir, "mndwi_s2_10m.tif"), mndwi, prof10)
    # debug masks (opțional)
    write_tif(os.path.join(args.out_dir, "valid_mask_10m.tif"), valid.astype("float32"), prof10)
    write_tif(os.path.join(args.out_dir, "water_mask_10m.tif"), water.astype("float32"), prof10)

if __name__ == "__main__":
    main()
