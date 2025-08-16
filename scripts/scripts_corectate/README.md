
# UHI Pipeline — Cleaned S2/L8 & SUHI

This mini-pipeline fixes common pitfalls that lead to wrong-looking maps (e.g., rivers lighting up hot):
- Correct scaling for **Landsat 8/9 L2** Surface Reflectance and Surface Temperature
- Correct scaling for **Sentinel-2 L2A** reflectance
- Proper **cloud/snow/water masking** (S2 SCL classes & L8 QA_PIXEL bits)
- Consistent resampling and alignment (20 m -> 10 m with bilinear for reflectance; nearest for masks)
- Robust regression from S2 indices to LST to create **10 m LST predictions** and a **SUHI** anomaly map

## Steps

1. **Sentinel-2 indices (10 m)**
```bash
python sentinel2_indices.py   --b04 /path/B04_10m.jp2   --b08 /path/B08_10m.jp2   --b03 /path/B03_10m.jp2   --b11 /path/B11_20m.jp2   --scl /path/SCL_20m.jp2   --out_dir output_s2/
```

2. **Landsat 8/9 L2 products (30 m)**
```bash
python landsat8_l2_products.py   --sr_b4 /path/SR_B4.tif   --sr_b5 /path/SR_B5.tif   --sr_b6 /path/SR_B6.tif   --st_b10 /path/ST_B10.tif   --qa_pixel /path/QA_PIXEL.tif   --out_dir output_l8/
```

3. **Fit + Predict LST at 10 m & SUHI**
```bash
python uhi_fit_predict.py   --s2_ndvi output_s2/ndvi_s2_10m.tif   --s2_ndbi output_s2/ndbi_s2_10m.tif   --s2_ndwi output_s2/ndwi_s2_10m.tif   --l8_lst output_l8/lst_l8_celsius_30m.tif   --out_dir output_uhi/
```

The script saves:
- `lst_pred_from_s2_10m.tif`: predicted LST in °C at 10 m
- `suhi_pred_10m.tif`: LST anomaly vs rural baseline (°C)
- `uhi_fit_report.txt`: R², MAE, coefficients; use it to sanity-check correlations (NDBI positive, NDVI negative).

## Why your rivers looked hot
If ST/reflectance scaling or masks are wrong, water often ends up with out-of-range values and stretches the color ramp, making rivers appear “hot”. We explicitly mask S2 `SCL==6` and/or `NDWI>0`, and for Landsat we mask `QA_PIXEL` water `bit7=1` along with clouds/shadows/snow.

## Notes
- Keep acquisitions **close in time** (±1–2 days) for good S2→L8 regression.
- If you already have water masks from other sources (e.g., OSM), you can multiply them into the masks.
- If memory is tight, modify the scripts to window-read in tiles; all operations are rasterio-compatible with windowing.

