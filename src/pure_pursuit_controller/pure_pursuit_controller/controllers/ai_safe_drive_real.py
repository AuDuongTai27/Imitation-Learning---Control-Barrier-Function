#!/usr/bin/env python3
"""
ai_safe_drive_real.py
─────────────────────
ROS 2 Node running AI inference (PyTorch) with Pure Pursuit expert override on real vehicle.
"""

import os
import math
import csv
import time
import json
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from tf2_ros import Buffer, TransformListener

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class DAggerMLP(nn.Module):
    def __init__(self, input_dim=60, output_dim=2, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class AiSafeDriveRealNode(Node):
    def __init__(self):
        super().__init__('ai_safe_drive_real_node')

        current_dir = os.path.dirname(os.path.abspath(__file__))

        # --- Parameters ---
        self.declare_parameter('model_path', os.path.join(current_dir, 'combine_1.pth'))
        self.declare_parameter('target_beams', 60)
        self.declare_parameter('ai_speed', 1.0)
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('max_steering_ai', 0.35)

        self.declare_parameter('waypoint_path', '/home/fablab/Desktop/f1tenth_waypoint.csv')
        self.declare_parameter('lookahead_dist', 1.0)
        self.declare_parameter('wheelbase', 0.39)
        self.declare_parameter('expert_speed', 1.0)
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')

        self.declare_parameter('override_threshold', 0.15)
        self.declare_parameter('override_hold_secs', 1.0)
        self.declare_parameter('drive_topic', '/drive')

        self.model_path        = self.get_parameter('model_path').value
        self.target_beams      = self.get_parameter('target_beams').value
        self.ai_speed          = self.get_parameter('ai_speed').value
        self.max_range         = self.get_parameter('max_range').value
        self.max_steer_ai      = self.get_parameter('max_steering_ai').value

        self.csv_path          = self.get_parameter('waypoint_path').value
        self.L                 = self.get_parameter('lookahead_dist').value
        self.wheelbase         = self.get_parameter('wheelbase').value
        self.expert_speed      = self.get_parameter('expert_speed').value
        self.odom_topic        = self.get_parameter('odom_topic').value
        self.map_frame         = self.get_parameter('map_frame').value
        self.base_frame        = self.get_parameter('base_frame').value

        self.override_threshold = self.get_parameter('override_threshold').value
        self.override_hold_secs = self.get_parameter('override_hold_secs').value
        self.drive_topic       = self.get_parameter('drive_topic').value

        # --- State ---
        self.lock = threading.Lock()
        self.latest_expert_steer = 0.0
        self.latest_expert_time  = 0.0

        self.override_active    = False
        self.override_until     = 0.0

        self.waypoints = self.load_waypoints(self.csv_path)
        self.last_idx  = 0

        # --- Load AI model ---
        self.model        = None
        self.target_mean  = None
        self.target_std   = None
        self.device       = None

        norm_path = os.path.splitext(self.model_path)[0] + '_norm.json'
        if os.path.exists(norm_path):
            try:
                with open(norm_path, 'r') as f:
                    stats = json.load(f)
                self.target_mean = np.array(stats['target_mean'], dtype=np.float32)
                self.target_std  = np.array(stats['target_std'],  dtype=np.float32)
                self.get_logger().info(f"Normalization stats loaded: mean={self.target_mean}, std={self.target_std}")
            except Exception as e:
                self.get_logger().warn(f"Cannot load norm stats: {e}")

        if _HAS_TORCH:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if not torch.cuda.is_available():
                torch.set_num_threads(1)
            if os.path.exists(self.model_path) and self.model_path.endswith('.pth'):
                try:
                    raw_model = DAggerMLP(input_dim=self.target_beams, output_dim=2, dropout=0.1).to(self.device)
                    raw_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                    raw_model.eval()
                    self.model = torch.jit.script(raw_model)
                    self.get_logger().info(f"AI model loaded & JIT-compiled from {self.model_path}")
                except Exception as e:
                    self.get_logger().error(f"Cannot load AI model: {e}")
            else:
                self.get_logger().error(f"Model file not found: {self.model_path}")
        else:
            self.get_logger().error("PyTorch not available!")

        # --- TF Buffer ---
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Pub / Sub ---
        self.drive_pub  = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/pure_pursuit/markers', 10)

        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self._log_startup()

    def scan_callback(self, msg: LaserScan):
        now = time.monotonic()

        ai_steer, ai_spd = self._run_ai(msg)

        with self.lock:
            expert_steer = self.latest_expert_steer
            expert_age   = now - self.latest_expert_time

        expert_valid = (expert_age < 0.5)

        if expert_valid:
            steer_diff = abs(ai_steer - expert_steer)
            if steer_diff > self.override_threshold:
                with self.lock:
                    self.override_active = True
                    self.override_until  = now + self.override_hold_secs
                self.get_logger().warn(
                    f"[OVERRIDE ON] diff={math.degrees(steer_diff):.1f}° AI={math.degrees(ai_steer):.1f}° Expert={math.degrees(expert_steer):.1f}°",
                    throttle_duration_sec=0.5
                )

        with self.lock:
            if self.override_active and now >= self.override_until:
                self.override_active = False
                self.get_logger().info("[OVERRIDE OFF] AI resumed control.")
            is_override = self.override_active

        if is_override and expert_valid:
            self._publish(expert_steer, self.expert_speed, mode='EXPERT')
        else:
            self._publish(ai_steer, ai_spd, mode='AI')

    def odom_callback(self, msg: Odometry):
        if len(self.waypoints) == 0:
            return

        x, y, yaw = self._get_pose(msg)
        target = self._get_target_point(x, y)
        steer  = self._calc_steering(target, x, y, yaw)

        with self.lock:
            self.latest_expert_steer = steer
            self.latest_expert_time  = time.monotonic()

        self._publish_markers(target, x, y)

    def _run_ai(self, scan_msg: LaserScan):
        if self.model is None or not _HAS_TORCH:
            return 0.0, 0.0

        scan = self._preprocess_scan(scan_msg)

        with torch.no_grad():
            x_norm   = torch.tensor(scan / 10.0, dtype=torch.float32).unsqueeze(0).to(self.device)
            output   = self.model(x_norm).cpu().squeeze(0).numpy()

        if self.target_mean is not None and self.target_std is not None:
            output = output * self.target_std + self.target_mean

        speed = float(np.clip(output[0], 0.0, self.ai_speed))
        steer = float(np.clip(output[1], -self.max_steer_ai, self.max_steer_ai))
        return steer, speed

    def _preprocess_scan(self, msg: LaserScan) -> np.ndarray:
        ranges     = np.array(msg.ranges, dtype=np.float32)
        crop_limit = math.radians(60.0)
        angles     = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min
        mask       = (angles >= -crop_limit) & (angles <= crop_limit)

        if not np.any(mask):
            return np.full(self.target_beams, self.max_range, dtype=np.float32)

        vr = ranges[mask]
        va = angles[mask]
        vr = np.where(np.isnan(vr) | np.isinf(vr), self.max_range, vr)
        vr = np.clip(vr, 0.0, self.max_range)
        return np.interp(
            np.linspace(-crop_limit, crop_limit, self.target_beams), va, vr
        ).astype(np.float32)

    def _get_pose(self, odom_msg):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
            x   = tf.transform.translation.x
            y   = tf.transform.translation.y
            q   = tf.transform.rotation
            yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        except Exception:
            x   = odom_msg.pose.pose.position.x
            y   = odom_msg.pose.pose.position.y
            q   = odom_msg.pose.pose.orientation
            yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        return x, y, yaw

    def _get_target_point(self, x, y):
        n       = len(self.waypoints)
        indices = [(self.last_idx + i) % n for i in range(50)]
        pts     = self.waypoints[indices]
        dists   = np.linalg.norm(pts - np.array([x, y]), axis=1)
        nearest = indices[int(np.argmin(dists))]
        self.last_idx = nearest

        idx = nearest
        for _ in range(n):
            idx = (idx + 1) % n
            if math.dist([x, y], self.waypoints[idx]) >= self.L:
                return self.waypoints[idx]
            if idx == nearest:
                break
        return self.waypoints[nearest]

    def _calc_steering(self, target, x, y, yaw):
        dx      = target[0] - x
        dy      = target[1] - y
        y_local = -dx * math.sin(yaw) + dy * math.cos(yaw)
        ld      = math.dist([x, y], target)
        return math.atan2(2 * self.wheelbase * y_local, ld ** 2)

    def load_waypoints(self, path):
        actual = path
        if not os.path.exists(actual):
            self.get_logger().warn(f"Waypoint not found: {path} — trying fallbacks...")
            home = os.path.expanduser('~')
            candidates = [
                os.path.join(home, 'f1_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv'),
                os.path.join(home, 'f1_ws/src/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv'),
            ]
            for c in candidates:
                if os.path.exists(c):
                    actual = c
                    self.get_logger().info(f"Found waypoints at: {c}")
                    break

        if not os.path.exists(actual):
            self.get_logger().error("No waypoint file found!")
            return np.array([])

        pts = []
        with open(actual) as f:
            reader = csv.reader(f)
            next(reader)
            for r in reader:
                pts.append([float(r[0]), float(r[1])])
        self.get_logger().info(f"Loaded {len(pts)} waypoints from {actual}")
        return np.array(pts)

    def _publish(self, steer: float, speed: float, mode: str):
        msg = AckermannDriveStamped()
        msg.header.stamp        = self.get_clock().now().to_msg()
        msg.header.frame_id     = 'laser'
        msg.drive.steering_angle = float(steer)
        msg.drive.speed          = float(speed)
        self.drive_pub.publish(msg)
        self.get_logger().info(
            f"[{mode:6s}] steer={math.degrees(steer):+6.1f}°  speed={speed:.2f} m/s",
            throttle_duration_sec=1.0)

    def _publish_markers(self, target, x, y):
        arr = MarkerArray()
        arr.markers.append(self._make_marker(0, target[0], target[1], 0.0, 1.0, 0.0))
        arr.markers.append(self._make_marker(1, x,          y,          0.0, 0.0, 1.0))
        self.marker_pub.publish(arr)

    def _make_marker(self, mid, x, y, r, g, b):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.id   = int(mid)
        m.type = Marker.SPHERE
        m.scale.x = m.scale.y = m.scale.z = 0.3
        m.color.a = 1.0
        m.color.r = float(r)
        m.color.g = float(g)
        m.color.b = float(b)
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        return m

    def _log_startup(self):
        self.get_logger().info("=" * 52)
        self.get_logger().info("  AI SAFE DRIVE (REAL VEHICLE) STARTED")
        self.get_logger().info(f"  Model      : {self.model_path}")
        self.get_logger().info(f"  Beams      : {self.target_beams}")
        self.get_logger().info(f"  AI speed   : {self.ai_speed} m/s")
        self.get_logger().info(f"  Expert spd : {self.expert_speed} m/s")
        self.get_logger().info(f"  Threshold  : {math.degrees(self.override_threshold):.1f}° ({self.override_threshold:.3f} rad)")
        self.get_logger().info(f"  Hold time  : {self.override_hold_secs} s")
        self.get_logger().info(f"  Waypoints  : {len(self.waypoints)} pts")
        self.get_logger().info("=" * 52)


def main(args=None):
    rclpy.init(args=args)
    node = AiSafeDriveRealNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Shutting down — stopping car.")
        node._publish(0.0, 0.0, 'STOP')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()�═══════
    #  STARTUP LOG
    # ══════════════════════════════════════════════════════════════════

    def _log_startup(self):
        self.get_logger().info("=" * 52)
        self.get_logger().info("  AI SAFE DRIVE (REAL VEHICLE) STARTED")
        self.get_logger().info(f"  Model      : {self.model_path}")
        self.get_logger().info(f"  Beams      : {self.target_beams}")
        self.get_logger().info(f"  AI speed   : {self.ai_speed} m/s")
        self.get_logger().info(f"  Expert spd : {self.expert_speed} m/s")
        self.get_logger().info(f"  Threshold  : {math.degrees(self.override_threshold):.1f}° "
                               f"({self.override_threshold:.3f} rad)")
        self.get_logger().info(f"  Hold time  : {self.override_hold_secs} s")
        self.get_logger().info(f"  Waypoints  : {len(self.waypoints)} pts")
        self.get_logger().info("=" * 52)


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = AiSafeDriveRealNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Shutting down — stopping car.")
        node._publish(0.0, 0.0, 'STOP')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()reprocess_scan(self, msg: LaserScan) -> np.ndarray:
        ranges     = np.array(msg.ranges, dtype=np.float32)
        crop_limit = math.radians(60.0)
        angles     = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min
        mask       = (angles >= -crop_limit) & (angles <= crop_limit)

        if not np.any(mask):
            return np.full(self.target_beams, self.max_range, dtype=np.float32)

        vr = ranges[mask]
        va = angles[mask]
        vr = np.where(np.isnan(vr) | np.isinf(vr), self.max_range, vr)
        vr = np.clip(vr, 0.0, self.max_range)
        return np.interp(
            np.linspace(-crop_limit, crop_limit, self.target_beams), va, vr
        ).astype(np.float32)

    def _publish(self, steer: float, speed: float, mode: str):
        msg = AckermannDriveStamped()
        msg.header.stamp        = self.get_clock().now().to_msg()
        msg.header.frame_id     = 'laser'
        msg.drive.steering_angle = float(steer)
        msg.drive.speed          = float(speed)
        self.drive_pub.publish(msg)

    def visualize(self, path_local, target_local):
        ma = MarkerArray()
        del_m = Marker(); del_m.action = Marker.DELETEALL; ma.markers.append(del_m)
        path_m = Marker(); path_m.header.frame_id = self.base_frame; path_m.id = 0; path_m.type = Marker.LINE_STRIP; path_m.action = Marker.ADD
        path_m.scale.x = 0.1; path_m.color.a = 1.0; path_m.color.g = 1.0; path_m.pose.orientation.w = 1.0
        for pt in path_local:
            p = Point(); p.x = pt[0]; p.y = pt[1]; p.z = 0.2; path_m.points.append(p)
        ma.markers.append(path_m)
        tgt_m = Marker(); tgt_m.header.frame_id = self.base_frame; tgt_m.id = 1; tgt_m.type = Marker.SPHERE; tgt_m.action = Marker.ADD
        tgt_m.pose.position.x = target_local[0]; tgt_m.pose.position.y = target_local[1]; tgt_m.pose.position.z = 0.3
        tgt_m.scale.x = 0.3; tgt_m.scale.y = 0.3; tgt_m.scale.z = 0.3; tgt_m.color.a = 1.0; tgt_m.color.r = 1.0; tgt_m.color.b = 0.0
        ma.markers.append(tgt_m)
        self.viz_pub.publish(ma)

    def _log_startup(self):
        self.get_logger().info("=" * 60)
        self.get_logger().info("  AI SAFE DRIVE WITH RRT* EXPERT (REAL VEHICLE) STARTED")
        self.get_logger().info("  [SELF-CONTAINED STANDALONE FILE WITH FALLBACK]")
        self.get_logger().info(f"  Model      : {self.model_path}")
        self.get_logger().info(f"  Beams      : {self.target_beams}")
        self.get_logger().info(f"  AI speed   : {self.ai_speed} m/s")
        self.get_logger().info(f"  Expert RRT*: {self.expert_speed} m/s (Obstacle Margin: 0.35m)")
        self.get_logger().info(f"  Threshold  : {math.degrees(self.override_threshold):.1f}° ({self.override_threshold:.3f} rad)")
        self.get_logger().info(f"  DAgger Save: {self.enable_dagger} -> {self.dataset_path}")
        self.get_logger().info(f"  Pre-Roll   : {self.pre_roll_secs} s history buffer")
        self.get_logger().info("=" * 60)


def main(args=None):
    rclpy.init(args=args)
    node = AiSafeDriveRealNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Shutting down — stopping car.")
        node._publish(0.0, 0.0, 'STOP')
        if node.enable_dagger and len(node.dagger_buffer) > 0:
            node._flush_dagger_buffer(node.dagger_buffer)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()