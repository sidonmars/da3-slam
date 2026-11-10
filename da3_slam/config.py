"""Run configuration: YAML with `_include`, dotted overrides, and the submap geometry."""

import json
import os

from omegaconf import OmegaConf

from da3_slam import ROOT

CFG_DIR = os.path.join(ROOT, 'cfg')


def _layer(path, _seen=None):
    _seen = _seen or set()
    assert path not in _seen, f'circular _include through {path!r}'
    _seen.add(path)
    cfg = OmegaConf.load(path)
    includes = cfg.pop('_include', []) or []
    out = OmegaConf.create({})
    for inc in includes:
        inc_path = inc if os.path.isabs(inc) else os.path.join(CFG_DIR, f'{inc}.yaml')
        out = OmegaConf.merge(out, _layer(inc_path, _seen))
    return OmegaConf.merge(out, cfg)


def parse_overrides(pairs):
    """['a.b=1', 'c=false'] -> nested dict with parsed scalars; later pairs win."""
    out = {}
    for p in pairs or []:
        k, v = p.split('=', 1)
        try:
            val = json.loads(v)
        except json.JSONDecodeError:
            val = {'true': True, 'false': False, 'null': None}.get(v.lower(), v)
        node = out
        parts = k.split('.')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = val
    return out


def _unknown_keys(cfg, overrides, prefix=''):
    out = []
    for k, v in overrides.items():
        path = f'{prefix}{k}'
        if k not in cfg:
            out.append(path)
        elif isinstance(v, dict) and OmegaConf.is_dict(cfg[k]):
            out += _unknown_keys(cfg[k], v, path + '.')
    return out


def load_config(path, overrides=None):
    cfg = _layer(os.path.abspath(path))
    if overrides:
        unknown = _unknown_keys(cfg, overrides)
        if unknown:
            raise ValueError(f'--override names keys the configuration does not have: '
                             f'{", ".join(unknown)}')
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def submap_geometry(cfg, fps=None):
    """Frame stride k from the frame rate, and the frame counts that follow from it:
    ``step`` new frames per submap, ``size`` frames per submap window."""
    b = cfg.backend
    k = int(b.frame_stride)
    if fps:
        rate = float(b.frame_rate)
        k = max(1, int(round(k * min(float(fps), rate) / rate)))
    b.frame_stride = k
    b.step = int(b.new_views) * k
    b.size = int(b.views) * k
    b.warmup_min = b.step
    cfg.frontend.max_gap = b.step
    return cfg
