# da3_slam

Visual SLAM built on Depth Anything 3 and a g2o Sim(3) pose graph. A small model (DA3-Small)
tracks every frame; a large one (DA3 Giant) builds 24-view submaps, which are aligned to each
other, loop-closed with DBoW2 retrieval and fused into one trajectory.

| Modality | KITTI odometry | EuRoC MAV |
|---|---|---|
| Monocular | yes | yes |
| Stereo (metric scale per submap) | yes | yes |
| Monocular + IMU | | yes |
| Stereo + IMU | | yes |

## Installation

Run from the folder that contains `da3_slam/`:

```bash
pip install -r da3_slam/requirements.txt
bash da3_slam/thirdparty/build.sh        # DBoW2 bindings, ALIKED ops
bash da3_slam/checkpoints/download.sh    # DA3 weights, ORB vocabulary, ALIKED weights
```

Datasets go in (or are linked into) `da3_slam/data/`:

```
data/kitti/sequences/<NN>/image_2, poses/<NN>.txt
data/euroc/<SEQ>/mav0/{cam0,cam1,imu0,state_groundtruth_estimate0}
```

If `import dpretrieval` fails with `GLIBCXX_3.4.32 not found`:
`export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`.

## Dependencies

- Python 3.10, PyTorch with CUDA
- `requirements.txt`: numpy, scipy, opencv-python, omegaconf, evo, g2o-python, huggingface_hub, ...
- Vendored in `thirdparty/`: Depth Anything 3, ALIKED, DBoW2, DPRetrieval, SALAD, MegaLoc
- Build tools for `thirdparty/build.sh`: cmake, a C++ compiler, the CUDA toolkit

## Usage

```bash
python -m da3_slam.main -c da3_slam/cfg/kitti.yaml --scenes 00 04
python -m da3_slam.main -c da3_slam/cfg/euroc.yaml --scenes MH_01_easy \
    --override sensor=stereo backend.imu.enable=true
python da3_slam/tests/test_da3_slam.py
```

`--override a.b=c` sets any key in `cfg/default.yaml`. Results go to `da3_slam/outputs/`.

## Layout

```
main.py            entry point: run a dataset, evaluate ATE
system.py          front-end + back-end behind one run()
frontend.py        per-frame tracking (DA3-Small)
mapping.py         back-end: submaps, graph, fusion, IMU initialization
submap.py          submap record
align.py           Sim(3) between two reconstructions of shared frames
alignment.py       edges and their information matrices
graph.py           g2o Sim(3) pose graph
longrange.py       long-context windows, loop verification
retrieval.py       loop candidates (DBoW2, SALAD, MegaLoc)
covis.py           ALIKED covisibility gate
refine.py          optional in-submap bundle adjustment
metric.py          stereo depth and per-submap metric scale
imu.py             IMU preintegration
imu_init.py        inertial-only initialization (scale, gravity, biases)
vi_graph.py        visual-inertial factor graph
vi_ba.py           full visual-inertial bundle adjustment
traversability.py  2D traversability map
geometry.py        Sim(3)/SE(3) helpers
model.py           Depth Anything 3 wrapper
config.py          YAML config with includes and overrides
stream.py          threaded live system
ros_worker.py      SLAM side of the ROS 2 bridge
ros_protocol.py    socket protocol of the bridge
cfg/               default.yaml, kitti.yaml, euroc.yaml
datasets/          KITTI and EuRoC loaders
eval/              ATE evaluation
ros2/              ROS 2 node, bag tools, replay script
scripts/           offline evaluation and replay tools
tests/             unit tests
thirdparty/        vendored dependencies and build.sh
checkpoints/       download.sh (weights are not committed)
```
