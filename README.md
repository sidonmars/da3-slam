# Dav3 front-end based SLAM

Visual SLAM built on Depth Anything 3 and a g2o Sim(3) pose graph. A small model (DA3-Small)
tracks every frame; a large one (DA3 Giant) builds 24-view submaps, which are aligned to each
other, loop-closed with DBoW2 retrieval and fused into one trajectory. The code is in
[da3_slam/](da3_slam/).

| Modality | KITTI odometry | EuRoC MAV |
|---|---|---|
| Monocular | yes | yes |
| Stereo (metric scale per submap) | yes | yes |
| Monocular + IMU | | yes |
| Stereo + IMU | | yes |

## Installation

```bash
conda create -n da3-slam python=3.10 -y
conda activate da3-slam
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # match your CUDA
pip install -r da3_slam/requirements.txt
bash da3_slam/thirdparty/build.sh          # DBoW2 bindings, ALIKED ops
bash da3_slam/checkpoints/download.sh      # DA3 weights, ORB vocabulary, ALIKED weights
```

Datasets go in (or are linked into) `da3_slam/data/`:

```
data/kitti/sequences/<NN>/image_2, poses/<NN>.txt
data/euroc/<SEQ>/mav0/{cam0,cam1,imu0,state_groundtruth_estimate0}
```

If `import dpretrieval` fails with `GLIBCXX_3.4.32 not found`:
`export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`.

## Usage

```bash
python -m da3_slam.main -c da3_slam/cfg/kitti.yaml --scenes 00 04
python -m da3_slam.main -c da3_slam/cfg/kitti.yaml --scenes 04 --override sensor=stereo
python -m da3_slam.main -c da3_slam/cfg/euroc.yaml --scenes MH_01_easy \
    --override sensor=stereo backend.imu.enable=true
```

- `--override a.b=c`: set any key in `da3_slam/cfg/default.yaml`
- `--scenes`: sequences to run; `--out`: output directory (default `da3_slam/outputs/`)
- `dataset.gray=true`: KITTI grayscale images (`image_0`/`image_1`)

Each run writes `results.json` and `summary.txt` (ATE per sequence) and `traj_<seq>.npz`.

## Dependencies

- Python 3.10, PyTorch with CUDA
- `da3_slam/requirements.txt`: numpy, scipy, opencv-python, omegaconf, evo, g2o-python, ...
- Vendored in `da3_slam/thirdparty/`: Depth Anything 3, ALIKED, DBoW2, DPRetrieval, SALAD, MegaLoc
- Build tools for `build.sh`: cmake, a C++ compiler, the CUDA toolkit

## Layout

See [da3_slam/README.md](da3_slam/README.md) for a file-by-file description.

```
da3_slam/
  main.py, system.py   entry point; front-end + back-end
  frontend.py          per-frame tracking
  mapping.py           submaps, pose graph, fusion, IMU initialization
  cfg/                 default, KITTI and EuRoC configs
  datasets/            KITTI and EuRoC loaders
  ros2/                ROS 2 bridge
  scripts/, tests/     evaluation tools, unit tests
  thirdparty/          vendored dependencies
```

## License

The packages under `da3_slam/thirdparty/` and the model weights keep their own licences.

## Acknowledgement

[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3),
[g2o](https://github.com/RainerKuemmerle/g2o), [DBoW2](https://github.com/dorian3d/DBoW2),
[DPV-SLAM](https://github.com/princeton-vl/DPVO), [SALAD](https://github.com/serizba/salad),
[MegaLoc](https://github.com/gmberton/MegaLoc), [ALIKED](https://github.com/Shiaoming/ALIKED),
[ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3), [evo](https://github.com/MichaelGrupp/evo).
