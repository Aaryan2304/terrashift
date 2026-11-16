import glob, os, sys
import xml.etree.ElementTree as ET  # trusted ESA Copernicus product metadata
import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.manifest import load_manifest

MANIFEST = load_manifest()
safes = sorted(p["path"] for p in MANIFEST["products"])

def loc(t):
    return t.split("}")[-1]

for s in safes:
    print("=" * 95)
    print(os.path.basename(s))
    root = ET.parse(os.path.join(s, "MTD_MSIL1C.xml")).getroot()

    # quantification + offsets
    print("  -- radiometric metadata --")
    for el in root.iter():
        t = loc(el.tag).upper()
        if t in ("QUANTIFICATION_VALUE", "RADIO_ADD_OFFSET", "RADIO_GAIN", "PHYSICAL_GAINS", "U"):
            print(f"     {loc(el.tag):22s} attrib={el.attrib} text={(el.text or '').strip()[:60]}")
    # offsets are usually inside a band list; print the parent structure
    print("  -- nodes named *OFFSET* with ancestry --")
    for el in root.iter():
        if "OFFSET" in loc(el.tag).upper():
            # walk up is not available; print siblings text
            print(f"     tag={loc(el.tag)} text={(el.text or '').strip()[:80]} attrib={el.attrib}")

    # spectral info resolution
    print("  -- spectral information --")
    for el in root.iter():
        if loc(el.tag) == "Spectral_Information":
            band = el.attrib.get("physicalBand")
            res = None
            for ch in el:
                if loc(ch.tag) == "RESOLUTION":
                    res = (ch.text or "").strip()
            print(f"     {band}: res={res}")

    # quality masks
    print("  -- QI_DATA masks --")
    qi = glob.glob(os.path.join(s, "GRANULE", "*", "QI_DATA", "MSK_*.jp2"))
    for q in sorted(qi):
        with rasterio.open(q) as ds:
            a = ds.read(1)
            vals, cnts = np.unique(a, return_counts=True)
            top = ", ".join(f"{v}:{c}" for v, c in list(zip(vals, cnts))[:8])
            print(f"     {os.path.basename(q):24s} shape={a.shape} dtype={a.dtype} "
                  f"min={a.min()} max={a.max()} unique={len(vals)} [{top}]")
    print()
