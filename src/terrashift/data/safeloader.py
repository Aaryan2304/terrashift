"""Sentinel-2 .SAFE product loader.

Reads MTD_MSIL1C.xml and the IMG_DATA/*.jp2 bands and converts DN to TOA reflectance as
``(DN + RADIO_ADD_OFFSET) / QUANTIFICATION_VALUE``, with DN 0 as NO_DATA. Offset and quantification
are read from the metadata, never hardcoded. Raster names (B01, B02) are reconciled with the
metadata names (B1, B2) and with IMAGE_FILE entries that omit the extension.
"""

import glob
import os
import xml.etree.ElementTree as ET  # trusted ESA Copernicus product metadata

import numpy as np
import rasterio
from rasterio.enums import Resampling

BAND_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
              "B08", "B8A", "B09", "B10", "B11", "B12"]


def _local(tag):
    """Strip the XML namespace from a tag name."""
    return tag.split("}")[-1]


def canonical_band(band):
    """Map a file band name to the metadata band name.

    B01 -> B1, B09 -> B9, B8A -> B8A, B11 -> B11.
    """
    b = band.upper()
    if b.startswith("B") and b != "B8A" and b[1:].isdigit():
        return "B" + str(int(b[1:]))
    return b


class SafeProduct:
    """One Sentinel-2 .SAFE product."""

    def __init__(self, safe_path):
        self.path = os.path.abspath(safe_path)
        self.name = os.path.basename(self.path)
        if not os.path.isdir(self.path):
            raise FileNotFoundError(self.path)

        meta_path = os.path.join(self.path, "MTD_MSIL1C.xml")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(meta_path)
        self.root = ET.parse(meta_path).getroot()

        self.quantification = self._read_quantification()
        self.band_id = self._read_band_id_map()
        self.offsets = self._read_offsets()
        self.resolution = self._read_resolution()
        self.wavelength = self._read_wavelengths()

        granules = [g for g in glob.glob(os.path.join(self.path, "GRANULE", "*"))
                    if os.path.isdir(g)]
        if not granules:
            raise FileNotFoundError("no GRANULE directory in " + self.path)
        self.granule = granules[0]
        img_dirs = glob.glob(os.path.join(self.granule, "IMG_DATA"))
        if not img_dirs:
            raise FileNotFoundError("no IMG_DATA directory in " + self.granule)
        self.img_dir = img_dirs[0]

        self.band_files = self._read_band_files()

    # ------------------------------------------------------------------ metadata
    def _read_quantification(self):
        for el in self.root.iter():
            if _local(el.tag).upper() == "QUANTIFICATION_VALUE":
                return float(el.text.strip())
        raise ValueError("QUANTIFICATION_VALUE not found in " + self.name)

    def _read_band_id_map(self):
        """Map canonical band name (B1, B8A, ...) to the metadata band_id."""
        mapping = {}
        for el in self.root.iter():
            if _local(el.tag) == "Spectral_Information":
                band = el.attrib.get("physicalBand")
                bid = el.attrib.get("bandId")
                if band and bid is not None:
                    mapping[canonical_band(band)] = int(bid)
        return mapping

    def _read_offsets(self):
        """Read RADIO_ADD_OFFSET per metadata band_id."""
        offsets = {}
        for el in self.root.iter():
            if _local(el.tag).upper() == "RADIO_ADD_OFFSET":
                bid = el.attrib.get("band_id")
                if bid is not None:
                    offsets[int(bid)] = float(el.text.strip())
        return offsets

    def _read_resolution(self):
        """Native spatial resolution in metres, per canonical band name."""
        res = {}
        for el in self.root.iter():
            if _local(el.tag) == "Spectral_Information":
                band = el.attrib.get("physicalBand")
                for ch in el:
                    if _local(ch.tag) == "RESOLUTION":
                        res[canonical_band(band)] = int(ch.text.strip())
        return res

    def _read_wavelengths(self):
        """Central wavelength in nm, per canonical band name."""
        wl = {}
        for el in self.root.iter():
            if _local(el.tag) == "Spectral_Information":
                band = el.attrib.get("physicalBand")
                for ch in el:
                    if _local(ch.tag) == "Wavelength":
                        for g in ch:
                            if _local(g.tag).upper() == "CENTRAL":
                                wl[canonical_band(band)] = float(g.text.strip())
        return wl

    def _read_band_files(self):
        """Map file band name (B01, ...) to the .jp2 path.

        The metadata IMAGE_FILE text omits the ".jp2" extension, so the file is
        located by glob inside IMG_DATA.
        """
        files = {}
        for el in self.root.iter():
            if _local(el.tag).upper() != "IMAGE_FILE":
                continue
            txt = (el.text or "").strip()
            base = os.path.basename(txt)
            for band in BAND_ORDER:
                if base.endswith("_" + band) or ("_" + band + ".") in base:
                    hits = glob.glob(os.path.join(self.img_dir, "*" + band + ".jp2"))
                    files[band] = hits[0] if hits else None
                    break
        # fill any missing band by a direct glob
        for band in BAND_ORDER:
            if band not in files or files[band] is None:
                hits = glob.glob(os.path.join(self.img_dir, "*" + band + ".jp2"))
                files[band] = hits[0] if hits else None
        return files

    # ------------------------------------------------------------------ helpers
    def offset_for(self, band):
        """Return the radiometric offset for a file band name."""
        bid = self.band_id[canonical_band(band)]
        return self.offsets[bid]

    def band_path(self, band):
        return self.band_files[band]

    # ------------------------------------------------------------------ raster I/O
    def read_band(self, band, window=None, out_shape=None,
                  resampling=Resampling.bilinear):
        """Read one band and convert it to TOA reflectance.

        Parameters
        ----------
        band : str
            File band name, for example "B04".
        window : tuple, optional
            ((row_start, row_end), (col_start, col_end)) in the band's own
            native pixel grid. Use this for windowed processing.
        out_shape : tuple, optional
            (rows, cols) of the output array. The band is resampled to this
            shape. Use this only when a common grid is needed.

        Returns
        -------
        reflectance : numpy.ndarray, float32
            TOA reflectance. Pixels where DN == 0 are NaN.
        transform : affine.Affine
            The geotransform of the returned array.
        crs : rasterio.crs.CRS
            The coordinate reference system.
        """
        path = self.band_files.get(band)
        if path is None:
            raise KeyError("band " + band + " not found in " + self.name)

        with rasterio.open(path) as ds:
            if window is not None:
                a = ds.read(1, window=window)
                transform = ds.window_transform(window)
            elif out_shape is not None:
                a = ds.read(1, out_shape=out_shape, resampling=resampling)
                transform = ds.transform * ds.transform.scale(
                    ds.width / out_shape[1], ds.height / out_shape[0])
            else:
                a = ds.read(1)
                transform = ds.transform
            crs = ds.crs

        a = a.astype(np.float32)
        reflectance = (a + self.offset_for(band)) / self.quantification
        reflectance[a == 0] = np.nan          # DN = 0 is NO_DATA
        return reflectance, transform, crs

    def read_band_native(self, band):
        """Read one band at its native resolution, no resampling."""
        return self.read_band(band)

    def summary(self):
        """A short text summary of the product."""
        lines = [self.name]
        lines.append("  quantification value : " + str(self.quantification))
        lines.append("  radiometric offset   : " + str(self.offsets))
        lines.append("  bands and resolution :")
        for band in BAND_ORDER:
            canon = canonical_band(band)
            if band in self.band_files and self.band_files[band]:
                lines.append("    " + band.ljust(4) + " " +
                             str(self.resolution.get(canon, "?")) + " m")
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    print(SafeProduct(path).summary())
