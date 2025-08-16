
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import os
import argparse

CLOUD_CLASSES = {3, 8, 9, 10}
BAD_CLASSES = {0, 1, 11}
WATER_CLASS = 6

def read_band(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile
    return data, profile

def resample_to(ref_profile, src_arr, src_profile, resampling):
    dst_arr = np.empty((ref_profile['height'], ref_profile['width']), dtype=np.float32)
    reproject(
        src_arr,
        dst_arr,
        src_transform=src_profile['transform'],
        src_crs=src_profile['crs'],
        dst_transform=ref_profile['transform'],
        dst_crs=ref_profile['crs'],
        dst_width=ref_profile['width'],
        dst_height=ref_profile['height'],
        resampling=resampling,
    )
    return dst_arr

def write_tif(path, arr, profile, nodata=np.nan):
    profile2 = profile.copy()
    profile2.update(dtype='float32', count=1, compress='deflate', nodata=nodata, driver='GTiff')
    with rasterio.open(path, 'w', **profile2) as dst:
        dst.write(arr.astype(np.float32), 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b04', required=True)
    ap.add_argument('--b08', required=True)
    ap.add_argument('--b03', required=True)
    ap.add_argument('--b11', required=True)
    ap.add_argument('--scl', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    red, prof10 = read_band(args.b04)
    nir, _ = read_band(args.b08)
    green, _ = read_band(args.b03)
    swir20, prof20 = read_band(args.b11)
    scl20, scl_prof = read_band(args.scl)

    # Convert DN -> reflectance: DN = 10000 * reflectance
    red /= 10000.0
    nir /= 10000.0
    green /= 10000.0

    # Resample 20 m bands to 10 m grid
    swir = resample_to(prof10, swir20, prof20, Resampling.bilinear) / 10000.0
    scl = resample_to(prof10, scl20, scl_prof, Resampling.nearest).astype(np.uint8)

    # Indices
    def safe_div(a, b):
        out = (a - b) / (a + b)
        out[(a + b) == 0] = np.nan
        return out

    ndvi = safe_div(nir, red)
    ndbi = safe_div(swir, nir)  # (SWIR - NIR)/(SWIR + NIR)
    ndwi = safe_div(green, nir) # McFeeters water index

    # Build mask
    bad = np.isin(scl, list(CLOUD_CLASSES | BAD_CLASSES)) | np.isnan(ndvi)
    water = (scl == WATER_CLASS) | (ndwi > 0.0)
    mask = ~(bad | water)

    for name, arr in [('NDVI', ndvi), ('NDBI', ndbi), ('NDWI', ndwi)]:
        out = np.where(mask, arr, np.nan).astype(np.float32)
        write_tif(os.path.join(args.out_dir, f'{name.lower()}_s2_10m.tif'), out, prof10)

    # Also export the mask for debugging
    write_tif(os.path.join(args.out_dir, 'valid_mask_s2_10m.tif'), mask.astype(np.float32), prof10, nodata=0.0)

if __name__ == '__main__':
    main()
