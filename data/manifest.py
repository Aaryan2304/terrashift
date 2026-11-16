"""Load the data manifest and resolve Sentinel-2 product paths.

``configs/data_manifest.yaml`` is the single source of truth for which .SAFE
products belong to which pair. Scripts must not hardcode product file names.

The manifest stores product paths relative to the ``SENTINEL_DATA_ROOT``
environment variable because the products are not copied into this repository.
``verify_manifest`` checks that every resolved path exists.
"""

import os
import re

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(ROOT, "configs", "data_manifest.yaml")


def load_manifest(path=None):
    """Read the manifest YAML. Returns a dict."""
    path = DEFAULT_MANIFEST if path is None else path
    with open(path) as fh:
        manifest = yaml.safe_load(fh)
    if not isinstance(manifest, dict) or "products" not in manifest:
        raise ValueError("manifest has no 'products' section: %s" % path)
    for product in manifest["products"]:
        value = product.get("path") if isinstance(product, dict) else None
        if not isinstance(value, str) or not value:
            raise ValueError("product %r has no valid path" %
                             (product.get("id") if isinstance(product, dict) else None))
        resolved = os.path.expanduser(os.path.expandvars(value))
        if re.search(r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[^}]+\})", resolved):
            raise ValueError(
                "unexpanded environment variable in product %r path; set SENTINEL_DATA_ROOT"
                % product.get("id"))
        product["path"] = resolved
    return manifest


def product_map(manifest):
    """Map product id -> product record."""
    return {p["id"]: p for p in manifest["products"]}


def pair_config(manifest, pair):
    """Return the manifest record for one pair, with resolved product paths.

    The result has: tile, crs, date1, date2, path1, path2.
    """
    pairs = manifest.get("pairs", {})
    if pair not in pairs:
        raise KeyError("pair %r not in manifest (have %s)" %
                       (pair, sorted(pairs)))
    cfg = dict(pairs[pair])
    products = product_map(manifest)
    for key in ("date1", "date2"):
        pid = cfg[key]
        if pid not in products:
            raise KeyError("product %r referenced by pair %r not in manifest" %
                           (pid, pair))
        cfg["path_" + key] = products[pid]["path"]
    return cfg


def pair_paths(manifest, pair):
    """Return (path_date1, path_date2) for a pair."""
    cfg = pair_config(manifest, pair)
    return cfg["path_date1"], cfg["path_date2"]


def all_pairs(manifest):
    return sorted(manifest.get("pairs", {}))


def verify_manifest(manifest):
    """Check that every product path exists. Returns a list of problems."""
    problems = []
    for product in manifest["products"]:
        path = product.get("path")
        if not path:
            problems.append("product %r has no path" % product.get("id"))
        elif not os.path.isdir(path):
            problems.append("product %r path does not exist: %s" %
                            (product.get("id"), path))
    for pair in all_pairs(manifest):
        try:
            pair_config(manifest, pair)
        except KeyError as exc:
            problems.append("pair %r: %s" % (pair, exc))
    return problems


if __name__ == "__main__":
    import sys
    manifest = load_manifest(sys.argv[1] if len(sys.argv) > 1 else None)
    problems = verify_manifest(manifest)
    for pair in all_pairs(manifest):
        cfg = pair_config(manifest, pair)
        print("pair %s tile %s crs %s" % (pair, cfg["tile"], cfg["crs"]))
        print("  date1 %s -> %s" % (cfg["date1"], cfg["path_date1"]))
        print("  date2 %s -> %s" % (cfg["date2"], cfg["path_date2"]))
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("\nmanifest OK: %d products, %d pairs" %
          (len(manifest["products"]), len(all_pairs(manifest))))
