"""YAML experiment configs: `inherit:` a parent file, then apply `key.sub=value` overrides."""

import copy
from pathlib import Path

import yaml


def deep_update(base, update):
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read(path):
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("inherit", None)
    cfg = _read(path.parent / parent) if parent else {}
    cfg = deep_update(cfg, raw)
    if "name" not in raw:
        cfg["name"] = path.stem
    return cfg


def set_dotted(cfg, dotted, value):
    node = cfg
    *parents, last = dotted.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[last] = value


def load_config(path, overrides=()):
    """Load a config; overrides look like 'replay.ratio=0.1' (values parsed as YAML)."""
    cfg = _read(path)
    for item in overrides:
        key, _, value = item.partition("=")
        set_dotted(cfg, key.strip(), yaml.safe_load(value))
    return cfg
