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

MSK_DETFOO (one per spectral band, 4 bits, 12 detector footprints) is a
detector footprint index, not a cloud mask. It is read but is not part of the
default validity decision.

Band order source: the Sentinel-2 Product Specifications Document, as
documented by the eoreader band-name reference,
https://eoreader.readthedocs.io/stable/api/eoreader.bands.band_names.S2MaskBandNames.html

Fail-closed rule
----------------
An earlier version padded a mask with all-zero bands when the file had fewer
bands than the specification requires. A zero flag means "valid", so a missing
band silently became "everything is valid". That is exactly the failure this
module must prevent. Missing, ambiguous, or unexpected quality information now
raises ``ValueError``.
"""

import glob
import math
import os

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds

from data.grids import transforms_close

CLASSI_BANDS = ["opaque_cloud", "cirrus", "snow_ice"]
QUALIT_BANDS = ["ancillary_lost", "ancillary_degraded", "msi_lost",
                "msi_degraded", "defective", "nodata", "partially_corrected",
                "saturated"]

CLASSI_FILE = "MSK_CLASSI_B00.jp2"

#: Every MSK_QUALIT bit that invalidates an observation. Bit 7 ("partially
#: corrected, cross-talk") is included because the correction is not complete;
#: the choice is recorded so it can be changed deliberately, not by accident.
DEFAULT_INVALID_QUALIT_FLAGS = tuple(QUALIT_BANDS)


def _require_unique(qi_dir, pattern):
    """Return the single file matching ``pattern``.

    Raises when the file is missing or when more than one file matches, so a
    product restructure cannot silently select the wrong mask.
    """
    hits = sorted(glob.glob(os.path.join(qi_dir, pattern)))
    if not hits:
        raise FileNotFoundError("%s not found in %s" % (pattern, qi_dir))
    if len(hits) > 1:
        raise ValueError("ambiguous quality mask %s: %d matches in %s"
                         % (pattern, len(hits), qi_dir))
    return hits[0]


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
        path = _require_unique(self.qi_dir, CLASSI_FILE)
        with rasterio.open(path) as ds:
            if ds.count != len(CLASSI_BANDS):
                raise ValueError(
                    "%s has %d bands; the product specification defines %d"
                    % (path, ds.count, len(CLASSI_BANDS)))
            bands = [ds.read(i).astype(np.uint8) for i in range(1, ds.count + 1)]
            return np.stack(bands), ds.transform, ds.crs

    def read_qualit(self, band):
        """Read the eight MSK_QUALIT bands for one spectral band.

        Returns (array, transform, crs) with shape (8, H, W) at the band's
        native resolution.
        """
        path = _require_unique(self.qi_dir, "MSK_QUALIT_%s.jp2" % band)
        with rasterio.open(path) as ds:
            if ds.count != len(QUALIT_BANDS):
                raise ValueError(
                    "%s has %d bands; the product specification defines %d"
                    % (path, ds.count, len(QUALIT_BANDS)))
            bands = [ds.read(i).astype(np.uint8) for i in range(1, ds.count + 1)]
            return np.stack(bands), ds.transform, ds.crs

    def read_detfoo(self, band):
        """Read the MSK_DETFOO detector-footprint mask for one spectral band."""
        path = _require_unique(self.qi_dir, "MSK_DETFOO_%s.jp2" % band)
        with rasterio.open(path) as ds:
            return ds.read(1).astype(np.uint8), ds.transform, ds.crs

    # ------------------------------------------------------------------ combined
    def _flags_on_grid(self, path, names, grid, detail, prefix):
        """OR every band of a quality file into a mask on ``grid``.

        Each band is read with nearest-neighbour decimation, then its resulting
        transform is checked against the analysis grid. No reprojection is
        applied here: the requirement is that the mask already shares the
        product's UTM grid, and a mismatch means the product is not what the
        caller assumed.
        """
        invalid = np.zeros((grid.height, grid.width), dtype=bool)
        with rasterio.open(path) as ds:
            if ds.count != len(names):
                raise ValueError(
                    "%s has %d bands; the product specification defines %d"
                    % (path, ds.count, len(names)))
            if ds.crs is not None and grid.crs is not None and \
                    CRS.from_user_input(ds.crs) != CRS.from_user_input(grid.crs):
                raise ValueError("%s CRS %s does not match the analysis grid %s"
                                 % (path, ds.crs, grid.crs))
            for index, name in enumerate(names, start=1):
                flag = ds.read(index, out_shape=(grid.height, grid.width),
                               resampling=Resampling.nearest)
                transform = ds.transform * ds.transform.scale(
                    ds.width / grid.width, ds.height / grid.height)
                if not transforms_close(transform, grid.transform):
                    raise ValueError(
                        "%s band %d does not land on the analysis grid "
                        "(transform %s vs %s)" % (path, index, transform, grid.transform))
                flag = flag.astype(bool)
                detail[prefix + name] = int(flag.sum())
                invalid |= flag
        return invalid

    def invalid_on_grid(self, grid, qualit_bands=(), classi=True,
                        qualit_flags=DEFAULT_INVALID_QUALIT_FLAGS):
        """Combined invalid-observation mask on an explicit analysis grid.

        Parameters
        ----------
        grid : data.grids.Grid
            Target grid. The masks are decimated or replicated with nearest
            sampling; a transform mismatch raises.
        qualit_bands : sequence of str
            Spectral bands whose MSK_QUALIT flags apply. Empty means skip
            MSK_QUALIT. Pass every band in the comparison contract to be
            conservative.
        classi : bool
            Include MSK_CLASSI (cloud, cirrus, snow/ice).
        qualit_flags : sequence of str
            Which MSK_QUALIT bits invalidate a pixel. Defaults to all eight.

        Returns
        -------
        invalid : numpy.ndarray, bool
        detail : dict
            Pixel counts per contributing flag plus ``total_invalid``.
        """
        selected = list(qualit_flags)
        unknown = set(selected) - set(QUALIT_BANDS)
        if unknown:
            raise ValueError("unknown MSK_QUALIT flags: %s" % sorted(unknown))

        invalid = np.zeros((grid.height, grid.width), dtype=bool)
        detail = {}

        if classi:
            path = _require_unique(self.qi_dir, CLASSI_FILE)
            invalid |= self._flags_on_grid(path, CLASSI_BANDS, grid,
                                           detail, prefix="classi_")

        for band in qualit_bands:
            path = _require_unique(self.qi_dir, "MSK_QUALIT_%s.jp2" % band)
            band_mask = np.zeros_like(invalid)
            band_detail = {}
            band_mask |= self._flags_on_grid(path, selected, grid,
                                             band_detail, prefix="")
            for name in selected:
                detail["qualit_%s_%s" % (band, name)] = band_detail[name]
            invalid |= band_mask

        detail["total_invalid"] = int(invalid.sum())
        detail["total_pixels"] = int(invalid.size)
        return invalid, detail

    @staticmethod
    def _flag_in_window(ds, index, transform, shape, crs):
        """One mask band, read through a window and put on the caller's grid."""
        left, top = transform * (0, 0)
        right, bottom = transform * (shape[1], shape[0])
        win = from_bounds(min(left, right), min(bottom, top),
                          max(left, right), max(bottom, top), ds.transform)
        # floor/ceil, not round: the source window must cover the destination
        # so that no edge row or column falls back to the fill value.
        col0 = max(0, int(math.floor(win.col_off)))
        row0 = max(0, int(math.floor(win.row_off)))
        col1 = min(ds.width, int(math.ceil(win.col_off + win.width)))
        row1 = min(ds.height, int(math.ceil(win.row_off + win.height)))
        if col1 <= col0 or row1 <= row0:
            raise ValueError("requested window lies outside %s" % ds.name)
        source_window = Window(col0, row0, col1 - col0, row1 - row0)
        source = ds.read(index, window=source_window)
        destination = np.zeros(shape, dtype=np.uint8)
        reproject(source=source, destination=destination,
                  src_transform=ds.window_transform(source_window),
                  src_crs=ds.crs, dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.nearest)
        return destination.astype(bool)

    def invalid_window(self, transform, shape, crs, qualit_bands=(),
                       classi=True, qualit_flags=DEFAULT_INVALID_QUALIT_FLAGS):
        """Invalid-observation mask for one window on an arbitrary grid.

        Same flags and the same fail-closed rules as :meth:`invalid_on_grid`,
        but each mask band is read through a window instead of being decimated
        over the whole tile. Use this when the caller works on a crop: the
        60 m MSK_CLASSI and the 10 m MSK_QUALIT are placed on the caller's grid
        with nearest sampling, so the mask lines up with the image window pixel
        for pixel.

        Parameters
        ----------
        transform, shape, crs
            The window's affine transform, ``(rows, cols)``, and CRS. These
            must be the transform and CRS that the image window was read with.
        qualit_bands : sequence of str
            Spectral bands whose MSK_QUALIT flags apply.
        """
        selected = list(qualit_flags)
        unknown = set(selected) - set(QUALIT_BANDS)
        if unknown:
            raise ValueError("unknown MSK_QUALIT flags: %s" % sorted(unknown))
        shape = (int(shape[0]), int(shape[1]))
        if shape[0] < 1 or shape[1] < 1:
            raise ValueError("window shape must be positive")
        if crs is None:
            raise ValueError("the window CRS is required to place the quality masks")

        invalid = np.zeros(shape, dtype=bool)
        detail = {}

        if classi:
            path = _require_unique(self.qi_dir, CLASSI_FILE)
            with rasterio.open(path) as ds:
                if ds.count != len(CLASSI_BANDS):
                    raise ValueError(
                        "%s has %d bands; the product specification defines %d"
                        % (path, ds.count, len(CLASSI_BANDS)))
                for index, name in enumerate(CLASSI_BANDS, start=1):
                    flag = self._flag_in_window(ds, index, transform, shape, crs)
                    detail["classi_" + name] = int(flag.sum())
                    invalid |= flag

        for band in qualit_bands:
            path = _require_unique(self.qi_dir, "MSK_QUALIT_%s.jp2" % band)
            with rasterio.open(path) as ds:
                if ds.count != len(QUALIT_BANDS):
                    raise ValueError(
                        "%s has %d bands; the product specification defines %d"
                        % (path, ds.count, len(QUALIT_BANDS)))
                for name in selected:
                    index = QUALIT_BANDS.index(name) + 1
                    flag = self._flag_in_window(ds, index, transform, shape, crs)
                    detail["qualit_%s_%s" % (band, name)] = int(flag.sum())
                    invalid |= flag

        detail["total_invalid"] = int(invalid.sum())
        detail["total_pixels"] = int(invalid.size)
        return invalid, detail

    def invalid_mask(self, reference_band="B08", out_shape=None):
        """Combined invalid mask on the reference band's own grid.

        Kept for the standalone audit path. ``invalid_on_grid`` is the entry
        point for the analysis pipeline because it takes an explicit grid.
        """
        ref_path = self.product.band_path(reference_band)
        with rasterio.open(ref_path) as ds:
            if out_shape is None:
                out_shape = (ds.height, ds.width)
            dst_transform = ds.transform * ds.transform.scale(
                ds.width / out_shape[1], ds.height / out_shape[0])
            from data.grids import Grid
            grid = Grid(None if ds.crs is None else CRS.from_user_input(ds.crs),
                        dst_transform, int(out_shape[1]), int(out_shape[0]))
        invalid, detail = self.invalid_on_grid(grid, qualit_bands=(reference_band,))
        return invalid, dst_transform, grid.crs, detail


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
