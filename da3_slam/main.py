"""Run da3_slam on a dataset.

    python -m da3_slam.main -c da3_slam/cfg/kitti.yaml [--scenes 00] [--override graph.weighting=info]
"""

import argparse
import json
import os
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from da3_slam import datasets
from da3_slam import eval as evaluation

from da3_slam import ROOT
from da3_slam.config import load_config, parse_overrides
from da3_slam.system import DA3SLAM

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('--override', nargs='*', default=None, help='dotted key=value pairs')
    ap.add_argument('--scenes', nargs='*', default=None, help='overrides dataset.scenes')
    ap.add_argument('--out', default=None, help='overrides output.dir')
    ap.add_argument('--quiet', action='store_true')
    return ap.parse_args()


def dataset_args(dcfg, model_name):
    """The argparse namespace `da3_slam.datasets` reads, from the config's dataset section."""
    cls = datasets.get(str(dcfg.name))
    root = str(dcfg.root)
    ns = SimpleNamespace(data_path=root if os.path.isabs(root) else os.path.join(ROOT, root),
                         scenes=list(dcfg.scenes or []) or None,
                         only=None, stride=dcfg.get('stride'), max_frames=dcfg.get('max_frames'),
                         res_hw=None)
    for flag, kw in cls.options:
        dest = kw.get('dest', flag.lstrip('-').replace('-', '_'))
        setattr(ns, dest, dcfg.get(dest, kw.get('default')))
    return cls(ns, model_name)


def online_ate(seq, online, out_dir, plot_mode):
    from da3_slam.eval.ate import _ate
    gt = np.asarray(seq.poses)
    est = online if seq.gt_index is None else online[np.asarray(seq.gt_index)]
    return float(_ate(gt, est, out_dir, f'{seq.name}_online', True, plot_mode, plot=False)[0])


def gravity_error(seq, poses):
    """Degrees between the estimate's -z and the ground truth's -z after aligning the two
    trajectories (valid for datasets whose ground-truth world is gravity-aligned)."""
    gt = np.asarray(seq.poses)
    est = poses if seq.gt_index is None else poses[np.asarray(seq.gt_index)]
    a, b = est[:, :3, 3], gt[:, :3, 3]
    A, B = a - a.mean(0), b - b.mean(0)
    U, _, Vt = np.linalg.svd(B.T @ A)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    z = R @ np.array([0.0, 0.0, -1.0])
    return float(np.degrees(np.arccos(np.clip(z @ np.array([0.0, 0.0, -1.0]), -1, 1))))


def main():
    args = arguments()
    overrides = parse_overrides(args.override)
    cfg = load_config(args.config, overrides)
    if args.scenes:
        cfg.dataset.scenes = list(args.scenes)
    if args.out:
        cfg.output.dir = args.out
    verbose = bool(cfg.output.verbose) and not args.quiet

    ds = dataset_args(cfg.dataset, str(cfg.model.name))
    resolution = ds.resolution()
    fps = cfg.dataset.get('fps') or ds.rate()
    modality = str(cfg.get('sensor', 'mono'))
    out_dir = cfg.output.dir or os.path.join(ROOT, 'outputs',
                                             f'da3_slam_{cfg.dataset.name}_{modality}')
    out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    system = DA3SLAM(cfg, fps=fps)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump({'resolved': OmegaConf.to_container(system.cfg, resolve=True),
                   'overrides': overrides, 'fps': fps}, f, indent=1)
    b = system.cfg.backend
    print(f'{ds.name} {modality} at {resolution[0]}x{resolution[1]} | size {b.size} step '
          f'{b.step} frame_stride {b.frame_stride} | graph {cfg.graph.weighting} -> {out_dir}',
          flush=True)

    res_path = os.path.join(out_dir, 'results.json')
    results = {}
    for seq in ds.sequences(resolution):
        n = seq.images.shape[1]
        print(f'\n=== {seq.name}: {n} frames ===', flush=True)
        try:
            torch.cuda.reset_peak_memory_stats()
            system.reset(calib_K=(torch.from_numpy(np.asarray(seq.K)).float()
                                  if cfg.dataset.get('calib') and seq.K is not None else None))
            ds.attach(seq, system.backend, system.cfg, modality, resolution)
            t0 = time.time()
            mem = system.run(seq.images, verbose=verbose)
            dt = time.time() - t0
            poses = mem.poses[:n].numpy()
            online = mem.online[:n].numpy()
            np.savez_compressed(
                os.path.join(out_dir, f"traj_{seq.name.replace('/', '_')}.npz"),
                pred=poses, online=online,
                gt=(np.zeros((0, 4, 4)) if seq.poses is None else np.asarray(seq.poses)),
                gt_index=(np.arange(0) if seq.gt_index is None else seq.gt_index),
                kf=np.array(mem.keyframes), submap_poses=mem.submap_poses.numpy())
            if mem.traversability is not None:
                from da3_slam.traversability import save_figure
                tag = seq.name.replace('/', '_')
                g_ = mem.traversability
                np.savez_compressed(os.path.join(out_dir, f'trav_{tag}.npz'),
                                    **{k: v for k, v in g_.items() if k != 'report'})
                save_figure(g_, os.path.join(out_dir, f'trav_{tag}.png'))
                torch.save({'locals': [sm.trav for sm in mem.submaps],
                            'S_t': mem.submap_poses.cpu(), 'poses': mem.poses.cpu(),
                            'frames': [sm.frames for sm in mem.submaps],
                            'submap_poses_local': [sm.poses.cpu() for sm in mem.submaps]},
                           os.path.join(out_dir, f'trav_local_{tag}.pt'))
                print(f"    traversability: {g_['report']}", flush=True)
            if cfg.graph.dump:
                tag = seq.name.replace('/', '_')
                system.backend.st.graph.save(os.path.join(out_dir, f'graph_{tag}.npz'))
                sms = mem.submaps
                np.savez(os.path.join(out_dir, f'submaps_{tag}.npz'),
                         frames=np.concatenate([np.asarray(sm.frames) for sm in sms]),
                         poses=np.concatenate([sm.poses.cpu().numpy() for sm in sms]),
                         poses_raw=np.concatenate([(sm.poses_raw if sm.poses_raw is not None
                                                    else sm.poses).cpu().numpy() for sm in sms]),
                         offsets=np.cumsum([0] + [len(sm) for sm in sms]),
                         start=np.array([sm.start for sm in sms]),
                         end=np.array([sm.end for sm in sms]), n=n)
            st = mem.stats
            rec = {'scene': seq.scene, 'frames': int(n), 'keyframes': len(mem.keyframes),
                   'time_s': dt, 'fps': n / max(dt, 1e-9), 'loops': int(st.get('n_loops', 0)),
                   'peak_mem_gb': torch.cuda.max_memory_allocated() / 1e9,
                   'n_edges': st.get('n_edges'), 'n_adjacent': st.get('n_adjacent'),
                   'n_span': st.get('n_span'), 'n_imu_rot': st.get('n_imu_rot'),
                   'n_metric': st.get('n_metric'), 'sensor': modality,
                   'gyro_bias': st.get('gyro_bias'), 'imu_init': st.get('imu_init'),
                   'n_long_context': st.get('n_long_context'), 'n_loop': st.get('n_loop'),
                   'refine': st.get('refine'), 'traversability': st.get('traversability'),
                   'loop_reject_reasons': st.get('loop_reject_reasons'),
                   'edges_rejected': st.get('edges_rejected'),
                   'graph': st.get('graph'), 'model_time_s': st.get('model_time_s')}
            if seq.poses is None:
                rec['no_gt'] = True
                print(f'  {seq.name}: no ground truth, trajectory saved', flush=True)
            else:
                rec.update(evaluation.evaluate(ds, seq, poses, mem.keyframes, out_dir))
                if cfg.output.save_online:
                    try:
                        rec['ate_online'] = online_ate(seq, online, out_dir, ds.plot_mode)
                    except Exception as e:                           # noqa: BLE001
                        print(f'  [warn] online ATE unavailable: {e}', flush=True)
                if rec.get('imu_init') and rec['imu_init'].get('applied'):
                    rec['gravity_err_deg'] = gravity_error(seq, poses)
                    print(f"    imu init: scale {rec['imu_init']['scale']:.4f} -> evo Sim(3) scale "
                          f"{rec.get('sim3_scale', float('nan')):.4f}, SE(3) ATE {rec.get('ate_se3')}, "
                          f"gravity error {rec['gravity_err_deg']:.2f} deg", flush=True)
                g = rec.get('graph') or {}
                print(f"--> {seq.name}: ATE={rec['ate']:.4f} kfATE={rec.get('ate_kf')} "
                      f"online={rec.get('ate_online')} fps={rec['fps']:.2f} loops={rec['loops']} "
                      f"edges={rec['n_edges']} chi2 {g.get('chi2_before', 0):.4g}->"
                      f"{g.get('chi2_after', 0):.4g} in {g.get('iters')} it, "
                      f"{1e3 * g.get('time_s', 0):.0f} ms", flush=True)
            results[seq.name] = rec
        except Exception as e:                                        # noqa: BLE001
            traceback.print_exc()
            results[seq.name] = {'scene': seq.scene, 'error': f'{type(e).__name__}: {e}'}
        finally:
            torch.cuda.empty_cache()
        with open(res_path, 'w') as f:
            json.dump(results, f, indent=1, default=float)

    text = evaluation.summary(ds, results, os.path.basename(os.path.normpath(out_dir)))
    print('\n' + text, flush=True)
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write(text + '\n')


if __name__ == '__main__':
    main()
