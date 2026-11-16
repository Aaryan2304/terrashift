"""Write a reproducibility manifest for one run.

The manifest answers: which code, which inputs, which packages, which
configuration, which outputs. It does not attempt to hash 0.5 GB of JPEG2000
imagery; it fingerprints the product metadata that determines the processing
(MTD_MSIL1C.xml) and records the size and count of the band files. That is
enough to tell whether the input changed without reading every band.

Usage:
    python scripts/make_run_manifest.py --output outputs/audit/run_manifest.json \
        --run-dir outputs/baseline_v2_observation_valid

Nothing is overwritten: an existing output path is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRACKED_PACKAGES = ("numpy", "scipy", "pandas", "rasterio", "torch", "onnx",
                    "onnxruntime", "shapely", "pyproj", "matplotlib", "PyYAML")


def sha256_file(path, limit=None):
    """Hex digest of a file, optionally of its first ``limit`` bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        remaining = limit
        while True:
            chunk = fh.read(1 << 20 if remaining is None else min(1 << 20, remaining))
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
                if remaining <= 0:
                    break
    return digest.hexdigest()


def git_state():
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
    }


def package_versions():
    from importlib import metadata

    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    versions["python"] = platform.python_version()
    versions["platform"] = platform.platform()
    return versions


def product_fingerprint(path):
    """Metadata hash plus band-file inventory for one .SAFE directory."""
    root = Path(path)
    entry = {"path": str(root), "exists": root.is_dir()}
    if not root.is_dir():
        return entry
    metadata = root / "MTD_MSIL1C.xml"
    entry["metadata_sha256"] = sha256_file(metadata) if metadata.is_file() else None
    bands = list(root.glob("GRANULE/*/IMG_DATA/*.jp2"))
    entry["band_file_count"] = len(bands)
    entry["band_total_bytes"] = sum(p.stat().st_size for p in bands)
    quality = list(root.glob("GRANULE/*/QI_DATA/*.jp2"))
    entry["quality_file_count"] = len(quality)
    entry["quality_total_bytes"] = sum(p.stat().st_size for p in quality)
    return entry


def directory_inventory(path):
    root = Path(path)
    if not root.is_dir():
        return None
    rows = []
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        row = {"path": str(item.relative_to(root)), "bytes": item.stat().st_size}
        # Small text artefacts are hashed; rasters are described by size only.
        if item.stat().st_size <= 5_000_000 and item.suffix.lower() in (
                ".json", ".csv", ".md", ".txt", ".yaml", ".yml", ".geojson"):
            row["sha256"] = sha256_file(item)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, action="append", default=None,
                        help="directory to inventory; repeatable")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/data_manifest.yaml")
    parser.add_argument("--note", default=None)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit("refusing to overwrite existing manifest: %s" % args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "packages": package_versions(),
        "config": {"path": str(args.config),
                   "sha256": sha256_file(args.config) if args.config.is_file() else None},
        "data_manifest": {"path": str(args.manifest),
                          "sha256": sha256_file(args.manifest)
                          if args.manifest.is_file() else None},
        "products": [],
        "runs": [],
        "note": args.note,
    }

    import yaml
    if args.manifest.is_file():
        manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
        variables = {"SENTINEL_DATA_ROOT": os.environ.get("SENTINEL_DATA_ROOT", "")}
        for product in manifest.get("products", []):
            raw = product.get("path", "")
            for key, value in variables.items():
                raw = raw.replace("${%s}" % key, value)
            payload["products"].append({
                "id": product.get("id"),
                "tile": product.get("tile"),
                "date": product.get("date"),
                "baseline": product.get("baseline"),
                **product_fingerprint(raw),
            })

    for run_dir in (args.run_dir or []):
        payload["runs"].append({"path": str(run_dir),
                                "files": directory_inventory(run_dir)})

    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.output)
    print("git commit:", payload["git"]["commit"], "dirty:", payload["git"]["dirty"])
    print("products:", len(payload["products"]), "run directories:", len(payload["runs"]))


if __name__ == "__main__":
    main()
