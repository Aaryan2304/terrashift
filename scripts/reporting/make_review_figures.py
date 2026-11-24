"""Render the two review figures from the authoritative reviewed-case records."""
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / "outputs/register/reviewed_cases_v2/reviewed_cases.geojson"
OUT = ROOT / "docs/figures"
COLUMNS = 4
CELL_W, CELL_H = 3.6, 2.3
WIDTH_IN = 14.4
WIDTH_PX = 1050


def load(source):
    data = json.loads(RECORDS.read_text(encoding="utf-8"))
    rows = [f["properties"] for f in data["features"] if f["properties"]["case_source"] == source]
    return rows


def render(rows, title, target):
    blocks = [(pair, [r for r in rows if r["pair_id"] == pair]) for pair in ("A", "B")]
    rows_per_block = [int(np.ceil(len(r) / COLUMNS)) for _, r in blocks]
    total = sum(rows_per_block)
    fig, axes = plt.subplots(total, COLUMNS, figsize=(WIDTH_IN, total * CELL_H))
    axes = np.atleast_2d(axes)
    for axis in axes.ravel():
        axis.set_axis_off()
    offset = 0
    for (pair, items), count in zip(blocks, rows_per_block):
        fig.text(0.01, 1 - (offset / total) - 0.005,
                 "Pair %s — %d reviewed locations" % (pair, len(items)),
                 fontsize=12, weight="bold", va="top")
        for position, record in enumerate(items):
            row, column = divmod(position, COLUMNS)
            axis = axes[offset + row][column]
            axis.imshow(mpimg.imread(record["overlay_path"]))
            axis.set_axis_off()
            axis.set_title("%s  (pair %s)" % (record["case_id"].replace("RC-", ""), record["pair_id"]),
                           fontsize=8)
        offset += count
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / target, dpi=WIDTH_PX / WIDTH_IN, format="jpg", pil_kwargs={"quality": 92})
    plt.close(fig)
    print("wrote", target, "with", len(rows), "records")


positive = load("positive_region_review")
negative = load("negative_region_review")
cp = collections.Counter(r["decision_status"] for r in positive)
cn = collections.Counter(r["decision_status"] for r in negative)
assert len(positive) == 26 and len(negative) == 12, (len(positive), len(negative))
render(positive, "TerraShift — %d reviewed locations: %d accepted, %d uncertain"
       % (len(positive), cp["accepted_man_made"], cp["uncertain"]), "f05_review_accepted.jpg")
render(negative, "TerraShift — %d reviewed locations: %d natural, %d non-target, %d uncertain"
       % (len(negative), cn["rejected_natural"], cn["rejected_non_target"], cn["uncertain"]),
       "f06_review_rejected.jpg")
