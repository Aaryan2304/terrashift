import os, glob, sys
import numpy as np
import rasterio
import cv2

from terrashift.data.manifest import load_manifest

MANIFEST = load_manifest()
PRODUCT_PATHS = {p["id"]: p["path"] for p in MANIFEST["products"]}


def img_dir(product_id):
    return glob.glob(os.path.join(PRODUCT_PATHS[product_id], "GRANULE", "*", "IMG_DATA"))[0]


def bp(product_id, band):
    return glob.glob(os.path.join(img_dir(product_id), f"*_{band}.jp2"))[0]


def read_win(product_id, band, r0, r1, c0, c1):
    with rasterio.open(bp(product_id, band)) as ds:
        return ds.read(1, window=((r0, r1), (c0, c1))).astype(np.float32)

# land windows (rows, cols) at native 10 m
WINDOWS = {
    "A": [(2000, 4000, 5000, 7000), (4000, 6000, 6000, 8000),
          (6000, 8000, 5000, 7000), (7000, 9000, 4000, 6000),
          (3000, 5000, 7000, 9000)],
    "B": [(2000, 4000, 2000, 4000), (4000, 6000, 4000, 6000),
          (6000, 8000, 6000, 8000), (3000, 5000, 7000, 9000),
          (5000, 7000, 2000, 4000)],
}

for pname, k1, k2 in [("A", "QBB_2022_12_23", "QBB_2023_05_12"),
                      ("B", "RGM_2023_10_06", "RGM_2026_10_22")]:
    print("=" * 90)
    print("PAIR", pname)
    s1, s2 = k1, k2
    shifts = []
    for (r0, r1, c0, c1) in WINDOWS[pname]:
        a = read_win(s1, "B08", r0, r1, c0, c1)
        b = read_win(s2, "B08", r0, r1, c0, c1)
        a = (a - a.mean()) / (a.std() + 1e-6)
        b = (b - b.mean()) / (b.std() + 1e-6)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_TRANSLATION,
                                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-7))
            dx, dy = warp[0, 2], warp[1, 2]
            shifts.append((dx, dy, cc))
            print(f"  win r{r0}-{r1} c{c0}-{c1}: dx={dx:+.3f}px dy={dy:+.3f}px "
                  f"({dx*10:+.2f}m,{dy*10:+.2f}m) corr={cc:+.3f}")
        except Exception as e:
            print(f"  win r{r0}-{r1} c{c0}-{c1}: ECC FAILED {e}")
    if shifts:
        arr = np.array(shifts)
        print(f"  median shift: dx={np.median(arr[:,0]):+.3f}px dy={np.median(arr[:,1]):+.3f}px "
              f"({np.median(arr[:,0])*10:+.2f}m,{np.median(arr[:,1])*10:+.2f}m)")
        print(f"  shift spread (std): dx={arr[:,0].std():.3f}px dy={arr[:,1].std():.3f}px")
        print(f"  corr: median={np.median(arr[:,2]):+.3f} min={arr[:,2].min():+.3f} max={arr[:,2].max():+.3f}")
        print(f"  windows with corr>0.5: {(arr[:,2]>0.5).sum()}/{len(arr)}")
