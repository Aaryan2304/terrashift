"""Paired geometric augmentation for bi-temporal change detection.

The same spatial transform must be applied to both dates and the label mask.
This module performs flips and quarter-turn rotations on CHW date tensors and
an HW mask. It uses PyTorch tensors only and adds no dependency.
"""

import torch


def apply_geometry(image1, image2, mask, k=0, flip_horizontal=False,
                   flip_vertical=False):
    """Apply one deterministic geometric transform to a date pair and mask."""
    if image1.ndim != 3 or image2.ndim != 3 or mask.ndim != 2:
        raise ValueError("images must be CHW and mask must be HW")
    if image1.shape[-2:] != image2.shape[-2:] or image1.shape[-2:] != mask.shape:
        raise ValueError("both image sizes and the mask size must match")
    if k not in (0, 1, 2, 3):
        raise ValueError("k must be 0, 1, 2 or 3 quarter-turns")

    if k:
        image1 = torch.rot90(image1, k, dims=(-2, -1))
        image2 = torch.rot90(image2, k, dims=(-2, -1))
        mask = torch.rot90(mask, k, dims=(-2, -1))
    if flip_horizontal:
        image1 = torch.flip(image1, dims=(-1,))
        image2 = torch.flip(image2, dims=(-1,))
        mask = torch.flip(mask, dims=(-1,))
    if flip_vertical:
        image1 = torch.flip(image1, dims=(-2,))
        image2 = torch.flip(image2, dims=(-2,))
        mask = torch.flip(mask, dims=(-2,))
    return image1, image2, mask


class PairedRandomGeometry:
    """Random flips and 90-degree rotations applied identically to a sample."""

    def __init__(self, p=0.5):
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.p = float(p)

    def __call__(self, sample):
        image1, image2, mask = sample["image1"], sample["image2"], sample["mask"]
        k = int(torch.randint(0, 4, ()).item()) if torch.rand(()) < self.p else 0
        flip_h = bool(torch.rand(()) < self.p)
        flip_v = bool(torch.rand(()) < self.p)
        image1, image2, mask = apply_geometry(
            image1, image2, mask, k=k,
            flip_horizontal=flip_h, flip_vertical=flip_v)
        result = dict(sample)
        result.update(image1=image1, image2=image2, mask=mask)
        valid_mask = sample.get("valid_mask")
        if valid_mask is not None:
            if valid_mask.ndim != 2 or valid_mask.shape != sample["mask"].shape:
                raise ValueError("valid_mask must be HW and match mask")
            if k:
                valid_mask = torch.rot90(valid_mask, k, dims=(-2, -1))
            if flip_h:
                valid_mask = torch.flip(valid_mask, dims=(-1,))
            if flip_v:
                valid_mask = torch.flip(valid_mask, dims=(-2,))
            result["valid_mask"] = valid_mask
        return result


if __name__ == "__main__":
    # Exhaustively test every discrete transform against a spatial marker.
    image1 = torch.arange(35, dtype=torch.float32).reshape(1, 5, 7)
    image2 = image1 + 100.0
    mask = (image1[0] == 9).to(torch.int64)
    checks = 0
    for k in range(4):
        for flip_h in (False, True):
            for flip_v in (False, True):
                a, b, y = apply_geometry(image1, image2, mask, k, flip_h, flip_v)
                assert torch.equal(b - a, torch.full_like(a, 100.0))
                assert torch.equal(y.bool(), a[0] == 9)
                assert a.shape[-2:] == y.shape
                checks += 1
    print("paired-transform combinations verified:", checks)

    # Check the random sample interface and preservation of non-image metadata.
    sample = {"image1": image1, "image2": image2, "mask": mask,
              "name": "toy", "split": "train"}
    transformed = PairedRandomGeometry(p=1.0)(sample)
    assert torch.equal(transformed["image2"] - transformed["image1"],
                       torch.full_like(transformed["image1"], 100.0))
    assert torch.equal(transformed["mask"].bool(), transformed["image1"][0] == 9)
    assert transformed["name"] == "toy" and transformed["split"] == "train"
    print("random paired sample transform and metadata preservation OK")
