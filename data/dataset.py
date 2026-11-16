"""PyTorch datasets for change detection.

OSCD100
-------
A subset of the Onera Satellite Change Detection dataset. 100 pre-cropped
256 x 256 pairs, 13 Sentinel-2 bands, binary change mask.

Structure on disk (after extraction)::

    data/oscd/extracted/Images/<split>_<id>/imgs_1_rect/B01.tif ... B8A.tif
    data/oscd/extracted/Images/<split>_<id>/imgs_2_rect/B01.tif ... B8A.tif
    data/oscd/extracted/Images/<split>_<id>/dates.txt
    data/oscd/extracted/<Split> Labels/<split>_<id>/cm/cm.png

The images are stored as uint16 digital numbers. The acquisition dates in
``dates.txt`` are 2015-08-01 and 2017-08-01, which predates processing
baseline 04.00 (October 2021). The absolute scaling is NOT verified from a
primary source. To avoid relying on it, the dataset normalises each band with
statistics computed on the training split.

Verified facts about OSCD100:

- 100 samples: 60 train, 20 val, 20 test.
- Every sample carries an identical ``dates.txt`` (20150801 / 20170801). The
  100 crops come from one acquisition pair, so the split is a random crop
  split and is not geographic.
- The rasters have no CRS and an identity transform.
- The dataset card says OSCD100 is "designed for tutorials and demonstrations,
  not benchmarking".

Licence: the original OSCD project page does not state one. The HuggingFace
OSCD100 mirror applies CC-BY-NC-SA-4.0. Treat the licence as unconfirmed.
Citation: Daudt, Le Saux, Boulch, Gousseau, IGARSS 2018.
"""

import os

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

BAND_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
              "B08", "B8A", "B09", "B10", "B11", "B12"]

DEFAULT_ROOT = "data/oscd/extracted"


def _read_band(path):
    with rasterio.open(path) as ds:
        return ds.read(1).astype(np.float32)


def list_samples(root=DEFAULT_ROOT):
    """Return a list of (split, sample_id, image1_dir, image2_dir, mask_path)."""
    images_root = os.path.join(root, "Images")
    out = []
    for split in ["train", "val", "test"]:
        label_root = os.path.join(root, "%s Labels" % split.capitalize())
        if not os.path.isdir(label_root):
            continue
        for sample in sorted(os.listdir(label_root)):
            sdir = os.path.join(label_root, sample)
            if not os.path.isdir(sdir):
                continue
            mask = os.path.join(sdir, "cm", "cm.png")
            imgs1 = os.path.join(images_root, sample, "imgs_1_rect")
            imgs2 = os.path.join(images_root, sample, "imgs_2_rect")
            if os.path.isfile(mask) and os.path.isdir(imgs1) and os.path.isdir(imgs2):
                out.append((split, sample, imgs1, imgs2, mask))
    return out


def read_pair(imgs1_dir, imgs2_dir, bands=BAND_ORDER):
    """Read one image pair as two [C, H, W] float32 arrays."""
    def read_dir(d):
        arrays = []
        for b in bands:
            path = os.path.join(d, b + ".tif")
            arrays.append(_read_band(path))
        return np.stack(arrays, axis=0)
    return read_dir(imgs1_dir), read_dir(imgs2_dir)


def read_mask(path):
    """Read a change mask. Returns [H, W] int64 with 0 = no change, 1 = change."""
    with rasterio.open(path) as ds:
        a = ds.read(1)
    return (a > 0).astype(np.int64)


def compute_statistics(samples, bands=BAND_ORDER):
    """Per-band mean and std over the given samples."""
    sums = np.zeros(len(bands), dtype=np.float64)
    sqsums = np.zeros(len(bands), dtype=np.float64)
    counts = 0
    for _, _, imgs1, imgs2, _ in samples:
        a1, a2 = read_pair(imgs1, imgs2, bands)
        for arr in (a1, a2):
            sums += arr.sum(axis=(1, 2))
            sqsums += (arr ** 2).sum(axis=(1, 2))
            counts += arr.shape[1] * arr.shape[2]
    mean = sums / max(counts, 1)
    var = sqsums / max(counts, 1) - mean ** 2
    std = np.sqrt(np.clip(var, 1e-12, None))
    return mean.astype(np.float32), std.astype(np.float32)


class OSCDDataset(Dataset):
    """OSCD100 change-detection dataset.

    Parameters
    ----------
    root : str
        Extraction root.
    split : str
        "train", "val" or "test".
    mean, std : array-like, optional
        Per-band normalisation statistics. Computed from the training split
        when not given.
    transform : callable, optional
        Applied to the sample dict.
    """

    def __init__(self, root=DEFAULT_ROOT, split="train", bands=BAND_ORDER,
                 mean=None, std=None, transform=None):
        self.root = root
        self.split = split
        self.bands = bands
        self.transform = transform
        self.samples = [s for s in list_samples(root) if s[0] == split]
        if not self.samples:
            raise RuntimeError("no samples for split %s in %s" % (split, root))
        if mean is None or std is None:
            train = [s for s in list_samples(root) if s[0] == "train"]
            mean, std = compute_statistics(train, bands)
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __len__(self):
        return len(self.samples)

    def change_fraction(self):
        """Fraction of changed pixels over the whole split, from masks only.

        This reads only the label PNGs, never the images, so it is cheap to
        call before training. Used to set the positive-class weight from the
        training split, which is a training-set statistic and leaks nothing.
        """
        changed = 0
        total = 0
        for _, _, _, _, mask_path in self.samples:
            mask = read_mask(mask_path)
            changed += int((mask > 0).sum())
            total += int(mask.size)
        if total == 0:
            return float("nan")
        return changed / total

    def __getitem__(self, index):
        _, sample, imgs1, imgs2, mask_path = self.samples[index]
        a1, a2 = read_pair(imgs1, imgs2, self.bands)
        mask = read_mask(mask_path)

        x1 = torch.from_numpy(a1)
        x2 = torch.from_numpy(a2)
        x1 = (x1 - self.mean) / self.std
        x2 = (x2 - self.mean) / self.std
        y = torch.from_numpy(mask)

        sample_dict = {"image1": x1, "image2": x2, "mask": y,
                       "name": sample, "split": self.split}
        if self.transform is not None:
            sample_dict = self.transform(sample_dict)
        return sample_dict


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        ds = OSCDDataset(split=split)
        print(split, len(ds), "samples; mean shape", tuple(ds.mean.shape))
        s = ds[0]
        print("  image1", tuple(s["image1"].shape), "image2", tuple(s["image2"].shape),
              "mask", tuple(s["mask"].shape),
              "change fraction %.4f" % float(s["mask"].float().mean()))
