"""Static INT8 quantisation support: calibration reader and QDQ options."""

import numpy as np
from onnxruntime.quantization import CalibrationDataReader


class CalibrationReader(CalibrationDataReader):
    """Feed calibration batches to the static quantiser."""

    def __init__(self, loader, input_names, limit):
        self.input_names = input_names
        self.batches = []
        for index, batch in enumerate(loader):
            if index >= limit:
                break
            self.batches.append({
                input_names[0]: batch["image1"].numpy().astype(np.float32),
                input_names[1]: batch["image2"].numpy().astype(np.float32)})
        if not self.batches:
            raise ValueError("no calibration batches were produced")
        self._position = 0

    def get_next(self):
        if self._position >= len(self.batches):
            return None
        batch = self.batches[self._position]
        self._position += 1
        return batch

    def rewind(self):
        self._position = 0
