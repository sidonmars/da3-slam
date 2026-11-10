#!/usr/bin/env bash
# End to end: KITTI sequence -> rosbag at camera rate -> node -> worker -> ROS outputs while it plays -> check.
#   bash da3_slam/ros2/replay_kitti.sh 04 [batch_run_dir] [worker flags, e.g. --lockstep]
set -uo pipefail
SEQ="${1:-04}"; BATCH="${2:-}"; FLAGS="${*:3}"
# Runs from the folder that holds da3_slam/, so that `python -m da3_slam...` imports it.
D=da3_slam; cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROS_ENV="${ROS_ENV:-$HOME/miniconda3/envs/ros_jazzy}"   # conda env with ROS 2 Jazzy
SLAM_PY="${SLAM_PY:-$HOME/miniconda3/bin/python}"        # python that runs da3_slam
OUT="$D/outputs/ros_kitti_$SEQ"; BAG="$OUT/bag"; SOCK="/tmp/da3_slam_$$.sock"
mkdir -p "$OUT"
ros () { env -u LD_PRELOAD -u PYTHONPATH ROS_DOMAIN_ID=73 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST bash -c "source $ROS_ENV/setup.bash >/dev/null 2>&1; $*"; }
[ -d "$BAG" ] || ros "python $D/ros2/kitti_to_bag.py $D/data/kitti $SEQ $BAG" || exit 1
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 $SLAM_PY -m da3_slam.ros_worker -c $D/cfg/kitti.yaml \
  --socket "$SOCK" --out "$OUT" --max-lag 0 $FLAGS --override dataset.calib=true traversability.enable=true \
  traversability.camera_height=1.65 > "$OUT/worker.log" 2>&1 & WORKER=$!
for _ in $(seq 240); do [ -S "$SOCK" ] && break; sleep 0.5; done
[ -S "$SOCK" ] || { echo "worker did not start"; tail "$OUT/worker.log"; exit 1; }
ros "python $D/ros2/collect_results.py $OUT/collected.npz" > "$OUT/collect.log" 2>&1 & COL=$!
ros "python $D/ros2/da3_slam_node.py --ros-args -p socket:=$SOCK -p exit_when_done:=true -p idle_timeout_s:=5.0 -p image_qos:=replay" > "$OUT/node.log" 2>&1 & NODE=$!
sleep 4
# the delay lets the topics match: early frames are lost otherwise
ros "ros2 bag play -d 5 $BAG" > "$OUT/play.log" 2>&1
wait $COL; C=$?; wait $NODE; N=$?; wait $WORKER; W=$?
echo "collector $C, node $N, worker $W"; tail -4 "$OUT/node.log"; cat "$OUT/collect.log"
env -u LD_PRELOAD $SLAM_PY $D/scripts/ros_replay_check.py "$OUT" "$OUT/collected.npz" "$SEQ" $BATCH
