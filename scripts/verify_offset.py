import os, glob, sys
import numpy as np
import rasterio
from rasterio.enums import Resampling

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.manifest import load_manifest
from preprocess.indices import normalized_difference

MANIFEST = load_manifest()
# product id -> absolute .SAFE path
PRODUCTS = {p["id"]: p["path"] for p in MANIFEST["products"]}
# Measured on all four products by scripts/audit.py. Do not assume these.
OFFSET = -1000.0
QUANT = 10000.0


def img_dir(path):
    return glob.glob(os.path.join(path, "GRANULE", "*", "IMG_DATA"))[0]


def bp(path, band):
    return glob.glob(os.path.join(img_dir(path), f"*_{band}.jp2"))[0]


def read_band(path, band, out_shape=(1098, 1098)):
    with rasterio.open(bp(path, band)) as ds:
        return ds.read(1, out_shape=out_shape, resampling=Resampling.nearest).astype(np.float32)


def read_window_10m(path, band, window_10m):
    """Read a window given in 10 m coordinates, scaled to the band's own grid.

    B11 is a 20 m band. Reading it with 10 m row/col numbers reads a four
    times larger area and gives the wrong mean.
    """
    r0, r1, c0, c1 = window_10m
    with rasterio.open(bp(path, band)) as ds:
        scale = 10.0 / float(ds.res[0])
        win = ((int(round(r0 * scale)), int(round(r1 * scale))),
               (int(round(c0 * scale)), int(round(c1 * scale))))
        return ds.read(1, window=win).astype(np.float32)

def refl(dn):
    return (dn + OFFSET) / QUANT

# sea / land windows in the QBB tile (native 10 m coordinates)
SEA = (1000, 2000, 1000, 2000)
LAND = (7000, 8000, 6000, 7000)

print("=" * 100)
print("EFFECT OF THE RADIO_ADD_OFFSET CORRECTION (offset = -1000, quant = 10000)")
print("=" * 100)
print(f"{'product':<18} {'window':<6} {'band':<5} {'raw DN':>9} {'reflectance':>12} {'NDVI_raw':>9} {'NDVI_refl':>10}")
for key, safe in PRODUCTS.items():
    for wname, (r0, r1, c0, c1) in [("SEA", SEA), ("LAND", LAND)]:
        vals = {}
        for b in ["B02", "B03", "B04", "B08", "B11"]:
            vals[b] = read_window_10m(safe, b, (r0, r1, c0, c1))
        # full-image indices at 100 m for the NDVI comparison
        b4 = read_band(safe, "B04"); b8 = read_band(safe, "B08")
        b3 = read_band(safe, "B03"); b11 = read_band(safe, "B11")
        ndvi_raw = normalized_difference(b8, b4)
        ndvi_ref = normalized_difference(refl(b8), refl(b4))
        ndwi_raw = normalized_difference(b3, b8)
        ndwi_ref = normalized_difference(refl(b3), refl(b8))
        print(f"{key:<18} {wname:<6} {'B08':<5} {vals['B08'].mean():>9.1f} {refl(vals['B08']).mean():>12.5f} "
              f"{ndvi_raw.mean():>9.4f} {ndvi_ref.mean():>10.4f}")
        print(f"{'':<18} {'':<6} {'B04':<5} {vals['B04'].mean():>9.1f} {refl(vals['B04']).mean():>12.5f} "
              f"{'NDWI_raw':>9} {'NDWI_refl':>10}")
        print(f"{'':<18} {'':<6} {'B03':<5} {vals['B03'].mean():>9.1f} {refl(vals['B03']).mean():>12.5f} "
              f"{ndwi_raw.mean():>9.4f} {ndwi_ref.mean():>10.4f}")

print()
print("=" * 100)
print("WATER AND BUILT-UP FRACTIONS: RAW DN vs CORRECTED REFLECTANCE")
print("=" * 100)
print(f"{'product':<18} {'water_raw':>10} {'water_refl':>11} {'built_raw':>10} {'built_refl':>11} {'NDVI_mean_refl':>15}")
for key, safe in PRODUCTS.items():
    b3 = read_band(safe, "B03"); b4 = read_band(safe, "B04")
    b8 = read_band(safe, "B08"); b11 = read_band(safe, "B11")
    ndwi_raw = normalized_difference(b3, b8)
    ndwi_ref = normalized_difference(refl(b3), refl(b8))
    ndvi_raw = normalized_difference(b8, b4)
    ndvi_ref = normalized_difference(refl(b8), refl(b4))
    ndbi_raw = normalized_difference(b11, b8)
    ndbi_ref = normalized_difference(refl(b11), refl(b8))
    water_raw = (ndwi_raw > 0.2).mean() * 100
    water_ref = (ndwi_ref > 0.2).mean() * 100
    built_raw = ((ndbi_raw > 0.02) & (ndvi_raw < 0.30) & (ndwi_raw < 0.15)).mean() * 100
    built_ref = ((ndbi_ref > 0.02) & (ndvi_ref < 0.30) & (ndwi_ref < 0.15)).mean() * 100
    print(f"{key:<18} {water_raw:>10.1f} {water_ref:>11.1f} {built_raw:>10.1f} {built_ref:>11.1f} {ndvi_ref.mean():>15.4f}")

print()
print("=" * 100)
print("PAIR A RADIOMETRY AFTER CORRECTION (same windows as the plan)")
print("=" * 100)
for wname, (r0, r1, c0, c1) in [("SEA", SEA), ("LAND", LAND)]:
    print(f"  window {wname}")
    for key in [p["id"] for p in MANIFEST["products"] if p["pair"] == "A"]:
        safe = PRODUCTS[key]
        row = []
        for b in ["B03", "B04", "B08", "B11"]:
            a = read_window_10m(safe, b, (r0, r1, c0, c1))
            row.append(f"{b}={refl(a).mean():.4f}")
        print(f"    {key}: " + "  ".join(row))
