import numpy as np


def evaluate(dataset, seq, poses, keyframes, out_dir):
    """Every metric ``dataset`` is scored by, for one sequence, as one flat dict."""
    out = {}
    for m in dataset.metrics:
        if m == 'ate':
            from da3_slam.eval.ate import evaluate as ate
            out.update(ate(seq, poses, keyframes, out_dir, dataset.plot_mode))
        else:
            raise ValueError(f'unknown metric {m!r}')
    return out


def summary(dataset, results, title):
    """The table a run ends with: ATE per sequence."""
    lines = [f'=== {title} ===',
             f"{'scene':<34} {'ATE':>8} {'ATEse3':>8} {'online':>8} {'scale':>7} {'fps':>7} "
             f"{'loops':>6} {'kf':>5}"]
    for name in sorted(results):
        r = results[name]
        if 'ate' not in r:
            if not r.get('no_gt'):
                lines.append(f"{name:<34} {r.get('error', 'FAILED')}")
            continue
        se3 = '-' if r.get('ate_se3') is None else f"{r['ate_se3']:.4f}"
        onl = '-' if r.get('ate_online') is None else f"{r['ate_online']:.4f}"
        sc = r.get('sim3_scale')
        sc = '-' if sc is None or not np.isfinite(sc) else f'{sc:.3f}'
        lines.append(f"{name:<34} {r['ate']:8.4f} {se3:>8} {onl:>8} {sc:>7} {r['fps']:7.2f} "
                     f"{r['loops']:6d} {r['keyframes']:5d}")
    ok = [r for r in results.values() if 'ate' in r]
    scored = [k for k, r in results.items() if not r.get('no_gt')]
    failed = sorted(k for k in scored if 'ate' not in results[k])
    if ok:
        se3s = [r['ate_se3'] for r in ok if r.get('ate_se3') is not None]
        label = 'MEAN' if not failed else f'MEAN ({len(ok)}/{len(scored)} ok)'
        lines.append(f"{label:<34} {np.mean([r['ate'] for r in ok]):8.4f} "
                     f"{(f'{np.mean(se3s):.4f}' if se3s else '-'):>8} {'':>8} {'':>7} "
                     f"{np.mean([r['fps'] for r in ok]):7.2f}")
    if failed:
        lines.append(f"FAILED ({len(failed)}): {', '.join(failed)}")
    return '\n'.join(lines)
