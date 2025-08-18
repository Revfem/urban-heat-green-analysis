
import argparse, numpy as np, rasterio
from rasterio.warp import reproject, Resampling

MODES = {
    "nearest": Resampling.nearest,   # pt. măști / categorice (ex: SCL)
    "bilinear": Resampling.bilinear, # pt. continue (LST, reflectanțe)
    "cubic": Resampling.cubic,
    "average": Resampling.average,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",  required=True, help="Rasterul de resamplat")
    ap.add_argument("--tmpl", required=True, help="Rasterul ȘABLON (grid/CRS țintă)")
    ap.add_argument("--dst",  required=True, help="Output GeoTIFF pe grila șablonului")
    ap.add_argument("--mode", default="bilinear", choices=MODES.keys())
    ap.add_argument("--dst_nodata", type=float, default=-9999)
    args = ap.parse_args()

    with rasterio.open(args.tmpl) as t, rasterio.open(args.src) as s:
        dst_prof = t.profile.copy()
        dst_prof.update(
            driver="GTiff", dtype="float32", count=1,
            nodata=args.dst_nodata, tiled=True, blockxsize=512, blockysize=512,
            compress="DEFLATE", BIGTIFF="IF_SAFER"
        )

        dst_arr = np.full((t.height, t.width), args.dst_nodata, dtype="float32")
        src_arr = s.read(1).astype("float32")

        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=s.transform, src_crs=s.crs,
            dst_transform=t.transform, dst_crs=t.crs,
            resampling=MODES[args.mode],
            src_nodata=s.nodata, dst_nodata=args.dst_nodata
        )

        with rasterio.open(args.dst, "w", **dst_prof) as out:
            out.write(dst_arr, 1)

if __name__ == "__main__":
    main()
