"""The ROS 2 side of the bridge. Subscribes to a camera, forwards each frame to the SLAM worker
and publishes what comes back while the stream runs:

    /da3_slam/pose            geometry_msgs/PoseStamped, per tracked frame, from the front-end
    /tf                       map -> da3_slam_camera, the same pose
    /da3_slam/path            nav_msgs/Path, latched: fused poses, renewed after each submap
    /da3_slam/traversability  nav_msgs/OccupancyGrid, latched: -1 unknown, 0 free, 100 obstacle,
                              renewed after each submap
    /da3_slam/status          std_msgs/String, latched: JSON, with final=true after the last update

The frame is ``map``, or ``da3_slam_world`` with traversability off. The map frame is
re-derived from all submaps at each update, so old poses move between updates.

The stream ends after ``idle_timeout_s`` without frames (0: only on Ctrl-C); then the worker
sends a final result. ``image_qos``: sensor (best effort, may drop, for a camera) or replay
(reliable, deep queue, for bags). Parameters: socket, image_topic, camera_info_topic,
idle_timeout_s, image_qos, max_rate_hz, exit_when_done.

Frames above ``max_rate_hz`` are dropped: the submap geometry is defined at 10 Hz, and a faster
camera would ask for submaps sooner than the giant model can build them.
"""

import array
import json
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from builtin_interfaces.msg import Time  # noqa: E402
from geometry_msgs.msg import PoseStamped, TransformStamped  # noqa: E402
from nav_msgs.msg import OccupancyGrid, Path  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,  # noqa: E402
                       qos_profile_sensor_data)
from rclpy.signals import SignalHandlerOptions  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from tf2_ros import TransformBroadcaster  # noqa: E402

from da3_slam import ros_protocol as proto  # noqa: E402


def stamp_msg(seconds):
    sec = int(seconds)
    return Time(sec=sec, nanosec=int(round((seconds - sec) * 1e9)) % 1000000000)


def fill_pose(ps_pose, T):
    ps_pose.position.x, ps_pose.position.y, ps_pose.position.z = (float(v) for v in T[:3, 3])
    o = ps_pose.orientation
    o.x, o.y, o.z, o.w = (float(v) for v in proto.quaternion_xyzw(T[:3, :3]))


class Da3SlamNode(Node):

    def __init__(self):
        super().__init__('da3_slam')
        for name, default in (('socket', '/tmp/da3_slam.sock'), ('image_topic', '/camera/image_raw'),
                              ('camera_info_topic', '/camera/camera_info'), ('idle_timeout_s', 0.0),
                              ('image_qos', 'sensor'), ('max_rate_hz', 10.0),
                              ('exit_when_done', False)):
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value
        self.sock = proto.connect(p('socket'))
        self.idle = float(p('idle_timeout_s'))
        self.min_dt = 0.9 / float(p('max_rate_hz')) if float(p('max_rate_hz')) > 0 else 0.0
        self.exit_when_done = bool(p('exit_when_done'))
        self.K, self.count, self.last, self.ended, self.done = None, 0, None, False, False
        self.stamps, self.updates = [], 0
        if p('image_qos') == 'replay':
            in_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2000,
                                reliability=ReliabilityPolicy.RELIABLE)
        elif p('image_qos') == 'sensor':
            in_qos = qos_profile_sensor_data
        else:
            raise ValueError(f'unknown image_qos {p("image_qos")!r}')
        latched = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(CameraInfo, p('camera_info_topic'), self.on_info, in_qos)
        self.create_subscription(Image, p('image_topic'), self.on_image, in_qos)
        self.pose_pub = self.create_publisher(PoseStamped, '/da3_slam/pose', 10)
        self.path_pub = self.create_publisher(Path, '/da3_slam/path', latched)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/da3_slam/traversability', latched)
        self.status_pub = self.create_publisher(String, '/da3_slam/status', latched)
        self.tf = TransformBroadcaster(self)
        self.create_timer(0.5, self.check_idle)
        threading.Thread(target=self.receive, daemon=True).start()
        self.get_logger().info(f'forwarding {p("image_topic")} to the worker')

    def on_info(self, msg):
        if self.K is None and any(msg.k):
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def on_image(self, msg):
        if self.ended:
            self.get_logger().warning('frame after the stream ended: ignored')
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.stamps:
            dt = stamp - self.stamps[-1]
            if 0 < dt < self.min_dt:
                return
            if dt <= 0:
                self.get_logger().warning(f'frame out of order by {-dt:.3f} s')
            elif len(self.stamps) > 8 and dt > 2.5 * float(np.median(np.diff(self.stamps))):
                self.get_logger().warning(f'gap of {dt:.3f} s in the image stream: frames lost?')
        self.stamps = self.stamps[-8:] + [stamp]
        arrays = {'rgb': proto.to_rgb8(msg.data, msg.encoding, msg.height, msg.width, msg.step)}
        if self.K is not None:
            arrays['K'] = self.K
        proto.send(self.sock, 'frame', {'stamp': stamp}, **arrays)
        self.count += 1
        self.last = self.get_clock().now()

    def check_idle(self):
        if self.idle <= 0 or self.ended or self.last is None:
            return
        if (self.get_clock().now() - self.last).nanoseconds * 1e-9 >= self.idle:
            self.end_stream('stream idle')

    def end_stream(self, why):
        if not self.ended:
            self.ended = True
            self.get_logger().info(f'{why}: {self.count} frames sent, waiting for the final result')
            proto.send(self.sock, 'end')

    def receive(self):
        """Everything the worker sends, until its final message."""
        try:
            while True:
                kind, meta, arrays = proto.recv(self.sock)
                if kind == 'pose':
                    self.publish_pose(meta, arrays['T'])
                elif kind == 'update':
                    self.publish_map(meta, arrays, final=False)
                elif kind == 'result':
                    self.publish_map(meta, arrays, final=True)
                    break
                else:
                    self.get_logger().error(f'worker failed: {meta}')
                    break
        except ConnectionError as e:
            self.get_logger().error(f'worker gone: {e}')
        self.done = True

    def publish_pose(self, meta, T):
        stamp = stamp_msg(meta['stamp'])
        ps = PoseStamped()
        ps.header.frame_id, ps.header.stamp = meta['frame_id'], stamp
        fill_pose(ps.pose, T)
        self.pose_pub.publish(ps)
        tf = TransformStamped()
        tf.header.frame_id, tf.header.stamp, tf.child_frame_id = meta['frame_id'], stamp, 'da3_slam_camera'
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = (
            float(v) for v in T[:3, 3])
        r = tf.transform.rotation
        r.x, r.y, r.z, r.w = (float(v) for v in proto.quaternion_xyzw(T[:3, :3]))
        self.tf.sendTransform(tf)

    def publish_map(self, meta, arrays, final):
        stamps = arrays['stamps']
        end = stamp_msg(float(stamps[-1]))
        has_map = 'poses_map' in arrays
        frame_id = 'map' if has_map else 'da3_slam_world'
        poses = arrays['poses_map'] if has_map else arrays['poses_world']
        path = Path()
        path.header.frame_id, path.header.stamp = frame_id, end
        for T, s in zip(poses, stamps):
            ps = PoseStamped()
            ps.header.frame_id, ps.header.stamp = frame_id, stamp_msg(float(s))
            fill_pose(ps.pose, T)
            path.poses.append(ps)
        self.path_pub.publish(path)

        if has_map:
            state, cell = arrays['state'], float(meta['report']['cell_m'])
            grid = OccupancyGrid()
            grid.header.frame_id, grid.header.stamp = 'map', end
            grid.info.map_load_time, grid.info.resolution = end, cell
            grid.info.width, grid.info.height = int(state.shape[0]), int(state.shape[1])
            grid.info.origin.position.x = float(arrays['origin_ij'][0] * cell)
            grid.info.origin.position.y = float(arrays['origin_ij'][1] * cell)
            grid.info.origin.orientation.w = 1.0
            grid.data = array.array('b', proto.occupancy_values(state).tobytes())
            self.grid_pub.publish(grid)
        self.updates += 1
        self.status_pub.publish(String(data=json.dumps(
            {'final': final, 'updates': self.updates, 'frames': len(poses),
             'submaps': meta.get('submaps'), 'report': meta.get('report')})))
        if final:
            self.get_logger().info(f'final: {len(poses)} poses after {self.updates} updates')


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Da3SlamNode()
    try:
        try:
            while rclpy.ok() and not (node.done and node.exit_when_done):
                rclpy.spin_once(node, timeout_sec=0.2)
        except KeyboardInterrupt:
            node.end_stream('Ctrl-C')
            while rclpy.ok() and not node.done:
                rclpy.spin_once(node, timeout_sec=0.2)
        if node.exit_when_done or node.done:   # let latched subscribers collect the results
            end = node.get_clock().now()
            while (node.get_clock().now() - end).nanoseconds < 5e9:
                rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
