"""Training configuration helpers shared by the training entry points."""


import numpy as np
import yaml


def load_config(path):
    """Read the YAML configuration."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError("config is not a mapping: %s" % path)
    return cfg

def resolve_pos_weight(configured, train_change_fraction):
    """Positive-class weight for BCEWithLogitsLoss.

    ``N_negative / N_positive``. When the config value is null it is computed
    from the training split, which is a training-set statistic and leaks
    nothing into validation.
    """
    if configured is not None:
        return float(configured)
    p = float(train_change_fraction)
    if not np.isfinite(p) or p <= 0.0:
        return None
    return (1.0 - p) / p

def parse_seeds(text, default_seed):
    """Resolve seeds from a CLI string or the configured seed list."""
    if text is None or text == "":
        values = default_seed if isinstance(default_seed, (list, tuple)) else [default_seed]
    elif isinstance(text, (list, tuple)):
        values = text
    else:
        values = [part.strip() for part in str(text).split(",") if part.strip()]
    seeds = [int(value) for value in values]
    if not seeds:
        raise ValueError("no seeds were configured")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed list contains duplicates: %s" % seeds)
    return seeds
