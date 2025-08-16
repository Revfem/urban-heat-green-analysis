
import rasterio
import numpy as np
import os
import argparse

# Usage:
#   python landsat8_l2_products.py \
#     --sr_b4 path/to/SR_B4.tif --sr_b5 path/to/SR_B5.tif --sr_b6 path/to/SR_B6.tif \
#     --st_b10 path/to/ST_B10.tif --qa_pixel path/to/QA_PIXEL.tif \
#     --out_dir output/

# Notes:
#   * Expects Landsat 8/9 Collection 2 Level-2 bands.
#   * Surface reflectance scale: reflectance = DN * 0.0000275 - 0.2 (C2). 
#   * Surface temperature: Kelvin = DN * 0.00341802 + 149.0; Celsius = K - 273.15.
#   * Masks clouds, cloud shadows, snow, water using QA_PIXEL bits.
#   * Outputs: NDVI, NDBI, LST_C (Celsius) as float32 GeoTIFFs + mask.

# QA_PIXEL bit positions for L8/9 C2 (LSDS docs / EE catalog)
BIT_FILL = 0
BIT_DILATED_CLOUD = 1
BIT_CIRRUS = 2
BIT_CLOUD = 3
BIT_CLOUD_SHADOW = 4
BIT_SNOW = 5
BIT_CLEAR = 6
BIT_WATER = 7

def read_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile
    return arr, profile

def write_tif(path, arr, profile, nodata=np.nan):
    p = profile.copy()
    p.update(dtype='float32', count=1, compress='deflate', nodata=nodata, driver='GTiff')
    with rasterio.open(path, 'w', **p) as dst:
        dst.write(arr.astype(np.float32), 1)

def bits_set(x, bit_idx):
    return ((x >> bit_idx) & 1).astype(bool)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sr_b4', required=True)
    ap.add_argument('--sr_b5', required=True)
    ap.add_argument('--sr_b6', required=True)
    ap.add_argument('--st_b10', required=True)
    ap.add_argument('--qa_pixel', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    red, profile = read_band(args.sr_b4)
    nir, _ = read_band(args.sr_b5)
    swir1, _ = read_band(args.sr_b6)
    st_dn, _ = read_band(args.st_b10)
    qa, _ = read_band(args.qa_pixel)

    qa = qa.astype('uint16')  # IMPORTANT: QA trebuie să fie întreg pt. operațiile pe biți

    # Scale to reflectance (C2)
    scale_reflectance = 0.0000275
    offset_reflectance = -0.2
    red = red * scale_reflectance + offset_reflectance
    nir = nir * scale_reflectance + offset_reflectance
    swir1 = swir1 * scale_reflectance + offset_reflectance

    # Surface Temperature in Kelvin & Celsius
    lst_K = st_dn * 0.00341802 + 149.0
    lst_C = lst_K - 273.15

    # Build mask
    clouds = bits_set(qa, BIT_CLOUD) | bits_set(qa, BIT_DILATED_CLOUD) | bits_set(qa, BIT_CIRRUS)
    shadow = bits_set(qa, BIT_CLOUD_SHADOW)
    snow = bits_set(qa, BIT_SNOW)
    water = bits_set(qa, BIT_WATER)
    fill = bits_set(qa, BIT_FILL)
    mask = ~(clouds | shadow | snow | water | fill)

    # Indices
    def safe_div(a, b):
        out = (a - b) / (a + b)
        out[(a + b) == 0] = np.nan
        return out

    ndvi = safe_div(nir, red)
    ndbi = safe_div(swir1, nir)

    ndvi = np.where(mask, ndvi, np.nan).astype(np.float32)
    ndbi = np.where(mask, ndbi, np.nan).astype(np.float32)
    lst_C = np.where(mask, lst_C, np.nan).astype(np.float32)

    write_tif(os.path.join(args.out_dir, 'ndvi_l8_30m.tif'), ndvi, profile)
    write_tif(os.path.join(args.out_dir, 'ndbi_l8_30m.tif'), ndbi, profile)
    write_tif(os.path.join(args.out_dir, 'lst_l8_celsius_30m.tif'), lst_C, profile)
    write_tif(os.path.join(args.out_dir, 'valid_mask_l8_30m.tif'), mask.astype(np.float32), profile, nodata=0.0)

if __name__ == '__main__':
    main()
