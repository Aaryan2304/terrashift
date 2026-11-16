import os, glob
import numpy as np
import rasterio
from rasterio.enums import Resampling

BASE = "/home/aaryan/projects/skyserve"
PRODUCTS = {
    "QBB_2022_12_23": "S2A_MSIL1C_20221223T054241_N0510_R005_T43QBB_20240809T025636.SAFE",
    "QBB_2023_05_12": "S2A_MSIL1C_20230512T053641_N0510_R005_T43QBB_20240908T023323.SAFE",
    "RGM_2023_10_06": "S2A_MSIL1C_20231006T052701_N0510_R105_T43RGM_20241101T065943.SAFE",
    "RGM_2026_10_22": "S2A_MSIL1C_20261022T053241_N0513_R105_T43RGM_20261022T091024.SAFE",
}
OFFSET = -1000.0
QUANT = 10000.0

def img_dir(safe):
    return glob.glob(os.path.join(BASE, safe, "GRANULE", "*", "IMG_DATA"))[0]

def bp(safe, band):
    return glob.glob(os.path.join(img_dir(safe), f"*_{band}.jp2"))[0]

def read_band(safe, band, out_shape=(1098, 1098)):
    with rasterio.open(bp(safe, band)) as ds:
        return ds.read(1, out_shape=out_shape, resampling=Resampling.nearest).astype(np.float32)

def refl(dn):
    return (dn + OFFSET) / QUANT

def sdiv(a, b, eps=1e-9):
    return np.divide(a, b, out=np.zeros_like(a), where=np.abs(b) > eps)

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
            with rasterio.open(bp(safe, b)) as ds:
                a = ds.read(1, window=((r0, r1), (c0, c1))).astype(np.float32)
            vals[b] = a
        # full-image indices at 100 m for the NDVI comparison
        b4 = read_band(safe, "B04"); b8 = read_band(safe, "B08")
        b3 = read_band(safe, "B03"); b11 = read_band(safe, "B11")
        ndvi_raw = sdiv(b8 - b4, b8 + b4)
        ndvi_ref = sdiv(refl(b8) - refl(b4), refl(b8) + refl(b4))
        ndwi_raw = sdiv(b3 - b8, b3 + b8)
        ndwi_ref = sdiv(refl(b3) - refl(b8), refl(b3) + refl(b8))
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
    ndwi_raw = sdiv(b3 - b8, b3 + b8)
    ndwi_ref = sdiv(refl(b3) - refl(b8), refl(b3) + refl(b8))
    ndvi_raw = sdiv(b8 - b4, b8 + b4)
    ndvi_ref = sdiv(refl(b8) - refl(b4), refl(b8) + refl(b4))
    ndbi_raw = sdiv(b11 - b8, b11 + b8)
    ndbi_ref = sdiv(refl(b11) - refl(b8), refl(b11) + refl(b8))
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
    for key in ["QBB_2022_12_23", "QBB_2023_05_12"]:
        safe = PRODUCTS[key]
        row = []
        for b in ["B03", "B04", "B08", "B11"]:
            with rasterio.open(bp(safe, b)) as ds:
                a = ds.read(1, window=((r0, r1), (c0, c1))).astype(np.float32)
            row.append(f"{b}={refl(a).mean():.4f}")
        print(f"    {key}: " + "  ".join(row))
