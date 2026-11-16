"""Sentinel-2 L1C quality masks.

The L1C product ships three groups of quality masks in GRANULE/<granule>/QI_DATA/.

MSK_CLASSI (60 m, 1830 x 1830, 3 single-bit bands):
    band 1 = opaque clouds
    band 2 = cirrus clouds
    band 3 = snow and ice

MSK_QUALIT (one file per spectral band, at that band's native resolution,
8 single-bit bands):
    band 1 = ancillary lost data
    band 2 = ancillary degraded data
    band 3 = MSI lost data
    band 4 = MSI degraded data
    band 5 = defective pixels
    band 6 = pixel nodata, inside the detectors
    band 7 = partially corrected, cross-talk
    band 8 = saturated pixels

MSK_DETFOO (one file per spectral band, 4 bits, 12 detector footprints).
This is a detector footprint mask, not a cloud mask.

Band order source: the Sentinel-2 Product Specification Document, as
documented by the eoreader band-name reference,
https://eoreader.readthedocs.io/stable/api/eoreader.bands.band_names.S2MaskBandNames.html
"""

import glob
import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

CLASSI_BANDS = ["opaque_cloud", "cirrus", "snow_ice"]
QUALIT_BANDS = ["ancillary_lost", "ancillary_degraded", "msi_lost",
                "msi_degraded", "defective", "nodata", "partially_corrected",
                "saturated"]


def _find(qi_dir, pattern):
    hits = sorted(glob.glob(os.path.join(qi_dir, pattern)))
    return hits[0] if hits else None


class QualityMasks:
    """Quality masks for one Sentinel-2 .SAFE product."""

    def __init__(self, product):
        self.product = product
        self.qi_dir = os.path.join(product.granule, "QI_DATA")
        if not os.path.isdir(self.qi_dir):
            raise FileNotFoundError(self.qi_dir)

    # ------------------------------------------------------------------ readers
    def read_classi(self):
        """Read the three MSK_CLASSI bands at 60 m.

        Returns (array, transform, crs) where array has shape (3, H, W)
        and the band order is opaque_cloud, cirrus, snow_ice.
        """
        path = _find(self.qi_dir, "MSK_CLASSI_B00.jp2")
        if path is None:
            raise FileNotFoundError("MSK_CLASSI_B00.jp2 not found in " + self.qi_dir)
        bands = []
        with rasterio.open(path) as ds:
            n = ds.count
            for i in range(1, n + 1):
                bands.append(ds.read(i).astype(np.uint8))
            transform = ds.transform
            crs = ds.crs
            out_shape = (ds.height, ds.width)
        # The product spec defines exactly three bands. Pad if a build ships fewer.
        while len(bands) < 3:
            bands.append(np.zeros(out_shape, dtype=np.uint8))
        return np.stack(bands[:3]), transform, crs

    def read_qualit(self, band):
        """Read the eight MSK_QUALIT bands for one spectral band.

        Returns (array, transform, crs) with shape (8, H, W) at the band's
        native resolution.
        """
        path = _find(self.qi_dir, "MSK_QUALIT_" + band + ".jp2")
        if path is None:
            raise FileNotFoundError("MSK_QUALIT_" + band + ".jp2 not found")
        bands = []
        with rasterio.open(path) as ds:
            for i in range(1, ds.count + 1):
                bands.append(ds.read(i).astype(np.uint8))
            transform = ds.transform
            crs = ds.crs
            out_shape = (ds.height, ds.width)
        while len(bands) < 8:
            bands.append(np.zeros(out_shape, dtype=np.uint8))
        return np.stack(bands[:8]), transform, crs

    def read_detfoo(self, band):
        """Read the MSK_DETFOO detector-footprint mask for one spectral band."""
        path = _find(self.qi_dir, "MSK_DETFOO_" + band + ".jp2")
        if path is None:
            raise FileNotFoundError("MSK_DETFOO_" + band + ".jp2 not found")
        with rasterio.open(path) as ds:
            return ds.read(1).astype(np.uint8), ds.transform, ds.crs

    # ------------------------------------------------------------------ combined
    @staticmethod
    def _resample(array, src_transform, src_crs, dst_transform, dst_shape,
                  crs, resampling=Resampling.nearest):
        """Resample a mask onto a target grid."""
        dst = np.zeros(dst_shape, dtype=array.dtype)
        reproject(source=array, destination=dst,
                  src_transform=src_transform, src_crs=src_crs,
                  dst_transform=dst_transform, dst_crs=crs,
                  resampling=resampling)
        return dst

    def invalid_mask(self, reference_band="B08", out_shape=None):
        """Build a combined invalid-observation mask on the reference band's grid.

        Invalid means: opaque cloud, cirrus, snow/ice, MSI lost, MSI degraded,
        defective, nodata or saturated.

        Parameters
        ----------
        reference_band : str
            The spectral band whose native grid defines the output grid.
        out_shape : tuple, optional
            (rows, cols) of the output. Defaults to the reference band's grid.

        Returns
        -------
        invalid : numpy.ndarray, bool
            True where the pixel is invalid.
        transform : affine.Affine
        crs : rasterio.crs.CRS
        detail : dict
            Pixel counts per contributing flag, on the output grid.
        """
        ref_path = self.product.band_path(reference_band)
        with rasterio.open(ref_path) as ds:
            if out_shape is None:
                out_shape = (ds.height, ds.width)
            dst_transform = ds.transform * ds.transform.scale(
                ds.width / out_shape[1], ds.height / out_shape[0])
            crs = ds.crs

        invalid = np.zeros(out_shape, dtype=bool)
        detail = {}

        # --- MSK_CLASSI (60 m) -> reference grid
        classi, cls_tr, cls_crs = self.read_classi()
        names = ["opaque_cloud", "cirrus", "snow_ice"]
        for i, name in enumerate(names):
            flag = self._resample(classi[i], cls_tr, cls_crs,
                                  dst_transform, out_shape, crs)
            flag = flag.astype(bool)
            detail[name] = int(flag.sum())
            invalid |= flag

        # --- MSK_QUALIT for the reference band (native resolution)
        qualit, q_tr, q_crs = self.read_qualit(reference_band)
        for i, name in enumerate(QUALIT_BANDS):
            flag = self._resample(qualit[i], q_tr, q_crs,
                                  dst_transform, out_shape, crs)
            flag = flag.astype(bool)
            detail["qualit_" + name] = int(flag.sum())
            invalid |= flag

        detail["total_invalid"] = int(invalid.sum())
        detail["total_pixels"] = int(invalid.size)
        return invalid, dst_transform, crs, detail


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.safeloader import SafeProduct

    prod = SafeProduct(sys.argv[1])
    qm = QualityMasks(prod)
    classi, tr, crs = qm.read_classi()
    print("MSK_CLASSI shape:", classi.shape)
    for i, name in enumerate(CLASSI_BANDS):
        print("  " + name.ljust(14) + " flagged pixels: " + str(int(classi[i].sum())))
    for band in ["B04", "B08", "B11"]:
        inv, _, _, detail = qm.invalid_mask(reference_band=band)
        print("invalid mask on " + band + " grid:", inv.shape,
              "invalid pixels:", detail["total_invalid"])
