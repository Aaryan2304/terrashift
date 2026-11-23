"""Model and session prediction for evaluation.

Returns per-pixel scores in the lattice ordering the accumulator expects.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from terrashift.data.validation_windows import IGNORE_VALUE
from terrashift.evaluation.lattice import LatticeAccumulator


def valid_of(batch):
    """Validity mask for a batch, defaulting to all-valid.

    The original-OSCD patches carry no validity mask: the rectified rasters have no
    NoData and the labels cover every pixel. The local SAFE dataset does carry one,
    so this helper keeps one evaluation path for both.
    """
    valid = batch.get("valid_mask")
    if valid is None:
        return np.ones(batch["mask"].shape, dtype=bool)
    return valid.numpy().astype(bool)

def session_probabilities(session, input_names, image1, image2):
    """Raw session output. The graph ends in sigmoid, so this is already a probability."""
    feed = {input_names[0]: np.ascontiguousarray(image1, dtype=np.float32),
            input_names[1]: np.ascontiguousarray(image2, dtype=np.float32)}
    output = np.asarray(session.run(None, feed)[0])
    if output.ndim == 4:
        output = output[:, 0]
    return output

def _accumulate_lattice(probabilities, batch, accumulator):
    valid = batch["valid_mask"].numpy()
    for index in range(probabilities.shape[0]):
        accumulator.add(int(batch["row_off"][index]), int(batch["col_off"][index]),
                        probabilities[index], valid[index],
                        core_offset=int(batch["core_offset"][index]),
                        core_count=int(batch["core_count"][index]))

def _finalise(accumulator, target):
    covered = (accumulator.count > 0) & (target != IGNORE_VALUE)
    scores = (accumulator.total[covered] / accumulator.count[covered]).astype(np.float32)
    labels = target[covered].astype(np.int64)
    return scores, labels

def predict_pair_onnx(session, input_names, entry, batch_size=4):
    """Per unique reviewed pixel scores from an ONNX session."""
    loader = DataLoader(entry["dataset"], batch_size=batch_size, shuffle=False,
                        num_workers=0)
    accumulator = LatticeAccumulator(entry["target"].shape, entry["col_offset"])
    for batch in loader:
        probabilities = session_probabilities(session, input_names,
                                              batch["image1"].numpy(),
                                              batch["image2"].numpy())
        _accumulate_lattice(probabilities, batch, accumulator)
    return _finalise(accumulator, entry["target"])

def predict_pair_torch(wrapper, entry, batch_size=4):
    """Per unique reviewed pixel scores from the torch wrapper.

    The wrapper already applies sigmoid, so its output is used unchanged.
    """
    loader = DataLoader(entry["dataset"], batch_size=batch_size, shuffle=False,
                        num_workers=0)
    accumulator = LatticeAccumulator(entry["target"].shape, entry["col_offset"])
    wrapper.eval()
    with torch.no_grad():
        for batch in loader:
            probabilities = wrapper(batch["image1"], batch["image2"]).squeeze(1).numpy()
            _accumulate_lattice(probabilities, batch, accumulator)
    return _finalise(accumulator, entry["target"])
