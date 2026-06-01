__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

"""Loader + small lexical-set accessor for the structural-rewrite config
(`raw_data/structural_rewrites.json`). Kept separate from the engine so the
predicate / primitive modules can read config without importing the engine."""

import json
import os

_CONFIG_CACHE = {}


def _default_config_path():
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "raw_data",
        "structural_rewrites.json",
    ))


def load_structural_rewrite_config(path=None):
    path = path or _default_config_path()
    if path in _CONFIG_CACHE:
        return _CONFIG_CACHE[path]
    with open(path, "r") as f:
        config = json.load(f)
    _CONFIG_CACHE[path] = config
    return config


def structural_lexical_set(name, path=None):
    config = load_structural_rewrite_config(path)
    return set(config.get("lexical_sets", {}).get(name, []))
