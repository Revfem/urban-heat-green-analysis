import numpy as np
import rasterio
import rasterio.mask
from shapely.geometry import Point, mapping, box
from shapely.ops import transform as shp_transform
from pyproj import Transformer, CRS
import matplotlib.pyplot as plt
import math

STACK = "output/stack_LSTday_NDVI_1km.tif"   # (band1=NDVI, band2=LST)
OUT_CLIP = "output/bucharest_stack_1km.tif"
OUT_UHMI = "output/bucharest_UHMI.tif"

# --- setări AOI ---
lon, lat = 26.1025, 44.4268           # București
radius_km = 120                        # ajustează

with rasterio.open(STACK) as src:
    crs_raster = src.crs
    transform = src.transform
    prof = src.profile
    # dimensiunea pixelului în unitățile CRS (metri sau grade)
    px_x = abs(transform.a)
    px_y = abs(transform.e)
    print("CRS:", crs_raster)
    print("Pixel size:", px_x, px_y)

# funcție utilitară: construiește o geometrie AOI corectă în CRS-ul rasterului
def make_aoi_in_raster_crs(crs_raster, lon, lat, radius_km):
    crs = CRS.from_user_input(crs_raster)
    raster_is_geographic = crs.is_geographic  # True => unități în grade, False => (de obicei) metri

    # transform coord. WGS84 -> CRS raster
    to_raster = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    pt_raster = shp_transform(to_raster, Point(lon, lat))

    if raster_is_geographic:
        # buffer în grade (aprox): 1° lat ~ 111 km, 1° lon ~ 111*cos(lat) km
        dlat = radius_km / 111.0
        dlon = radius_km / (111.0 * math.cos(math.radians(lat)))
        # construim o cutie (mai sigur decât un cerc distorsionat în grade)
        aoi_deg = box(lon - dlon, lat - dlat, lon + dlon, lat + dlat)
        aoi_raster = shp_transform(to_raster, aoi_deg)
    else:
        # CRS proiectat (metri): putem folosi un buffer în metri 👌
        aoi_raster = pt_raster.buffer(radius_km * 1000.0)

    return aoi_raster

aoi_raster = make_aoi_in_raster_crs(crs_raster, lon, lat, radius_km)

# --- decupare cu mask(crop=True) ---
with rasterio.open(STACK) as src:
    clipped, clipped_transform = rasterio.mask.mask(
        src, [mapping(aoi_raster)], crop=True, filled=True, nodata=np.nan
    )
    ndvi_c = clipped[0].astype(np.float32)
    lst_c  = clipped[1].astype(np.float32)

clip_prof = prof.copy()
clip_prof.update(
    height=ndvi_c.shape[0],
    width=ndvi_c.shape[1],
    transform=clipped_transform,
    count=2,
    dtype=rasterio.float32,
    nodata=np.nan,
    compress="DEFLATE",
)
with rasterio.open(OUT_CLIP, "w", **clip_prof) as dst:
    dst.write(ndvi_c, 1)
    dst.write(lst_c, 2)

# --- UHMI pe AOI ---
def normalize(arr):
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    return (arr - mn) / (mx - mn + 1e-6)

uhmi = normalize(lst_c) * (1 - normalize(ndvi_c))

uhmi_prof = clip_prof.copy()
uhmi_prof.update(count=1)
with rasterio.open(OUT_UHMI, "w", **uhmi_prof) as dst:
    dst.write(uhmi.astype(np.float32), 1)

plt.figure(figsize=(7.5, 5.5))
im = plt.imshow(uhmi, cmap="inferno")
plt.title(f"UHMI – Prioritate vegetație (București, r={radius_km} km)")
plt.colorbar(im, label="UHMI")
plt.axis("off")
plt.tight_layout()
plt.show()

print("Saved:", OUT_CLIP)
print("Saved:", OUT_UHMI)
