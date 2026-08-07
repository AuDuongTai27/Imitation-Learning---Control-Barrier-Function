#!/usr/bin/env python3
"""
dagger_inference_real_pytorch.py
────────────────────────────────
ROS 2 Node running PyTorch DAgger inference and dataset aggregation on real F1TENTH vehicle.
"""

import os
import csv
import math
import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class DAggerMLP(nn.Module):
    def __init__(self, input_dim=60, output_dim=2, dropout=0.1):
        super(DAggerMLP, self).__init__()
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


def resolve_model_path(path):
    if os.path.exists(path):
        return path
    dirname, filename = os.path.split(path)
    alt_path = os.path.join(dirname, 'models', filename)
    if os.path.exists(alt_path):
        return alt_path
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    alt_path2 = os.path.join(curr_dir, '..', 'models', filename)
    if os.path.exists(alt_path2):
        return os.path.abspath(alt_path2)
    return path


def resolve_dataset_path(path):
    if os.path.exists(path):
        return path
    dirname, filename = os.path.split(path)
    alt_path = os.path.join(dirname, 'datasets', filename)
    if os.path.exists(alt_path):
        return alt_path
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    alt_path2 = os.path.join(curr_dir, '..', 'datasets', filename)
    if os.path.exists(alt_path2):
        return os.path.abspath(alt_path2)
    return path


class DaggerInferenceRealPytorchNode(Node):
    def __init__(self):
        super().__init__('dagger_inference_real_pytorch_node')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_model = os.path.join(current_dir, '..', 'models', 'model_sim_5.pth')
        default_dataset = os.path.join(current_dir, '..', 'datasets', 'dagger_dataset_real.csv')

        # --- 1. Parameters ---
        self.declare_parameter('model_path', resolve_model_path(default_model))
        self.declare_parameter('dataset_path', resolve_dataset_path(default_dataset))
        self.declare_parameter('waypoint_path', '/home/adt/f1_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv')

        self.declare_parameter('target_beams', 60)
        self.declare_parameter('lookahead_dist', 1.0)
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('expert_speed', 1.0)        # Expert speed limit (m/s)
        self.declare_parameter('ai_speed', 1.0)            # AI speed limit (m/s)
        self.declare_parameter('cte_threshold', 0.15)      # CTE threshold for expert override (m)
        self.declare_parameter('buffer_size', 50)
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('max_steering_angle', 0.35)  # Max steering angle (rad)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')

        self.model_path     = resolve_model_path(self.get_parameter('model_path').value)
        self.dataset_path   = resolve_dataset_path(self.get_parameter('dataset_path').value)
        self.waypoint_path  = self.get_parameter('waypoint_path').value

        self.target_beams   = self.get_parameter('target_beams').value
        self.lookahead_dist = self.get_parameter('lookahead_dist').value
        self.wheelbase      = self.get_parameter('wheelbase').value
        self.expert_speed   = self.get_parameter('expert_speed').value
        self.ai_speed       = self.get_parameter('ai_speed').value
        self.cte_threshold  = self.get_parameter('cte_threshold').value
        self.buffer_size    = self.get_parameter('buffer_size').value
        self.max_range      = self.get_parameter('max_range').value
        self.max_steer      = self.get_parameter('max_steering_angle').value
        self.map_frame      = self.get_parameter('map_frame').value
        self.base_frame     = self.get_parameter('base_frame').value

        # --- 2. TF Buffer ---
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- 3. Load PyTorch Model ---
        self.model = None
        self.target_mean = None
        self.target_std = None

        norm_path = os.path.splitext(self.model_path)[0] + '_norm.json'
        if os.path.exists(norm_path):
            try:
                import json
                with open(norm_path, 'r') as f:
                    stats = json.load(f)
                self.target_mean = np.array(stats['target_mean'], dtype=np.float32)
                self.target_std = np.array(stats['target_std'], dtype=np.float32)
                self.get_logger().info(f"Loaded normalization stats from {norm_path}: mean={self.target_mean}, std={self.target_std}")
            except Exception as e:
                self.get_logger().warn(f"Failed to load normalization stats: {e}")

        if _HAS_TORCH:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if os.path.exists(self.model_path) and self.model_path.endswith('.pth'):
                try:
                    raw_model = DAggerMLP(input_dim=self.target_beams, output_dim=2, dropout=0.1).to(self.device)
                    raw_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                    raw_model.eval()
                    self.model = torch.jit.script(raw_model)
                    self.get_logger().info(f"Successfully loaded PyTorch model JIT from {self.model_path}")
                except Exception as e:
                    self.get_logger().error(f"Failed to load PyTorch weights: {e}")
            else:
                self.get_logger().error(f"Model .pth file not found: {self.model_path}")
        else:
            self.get_logger().error("PyTorch is not installed!")

        # --- 4. Waypoints & State ---
        self.waypoints = self.load_waypoints(self.waypoint_path)
        self.last_idx = 0

        self.car_x = 0.0
        self.car_y = 0.0
        self.car_yaw = 0.0
        self.odom_received = False

        self.buffer = []
        self.lock = threading.Lock()
        self.total_saved_samples = 0

        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
        if not os.path.exists(self.dataset_path):
            self._write_header()

        # --- 5. Pub/Sub ---
        odom_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, odom_qos)

        self.get_logger().info("==================================================")
        self.get_logger().info(" DAGGER REAL VEHICLE PYTORCH NODE READY")
        self.get_logger().info(f" Dataset Path: {self.dataset_path}")
        self.get_logger().info(f" CTE Threshold: {self.cte_threshold} m")
        self.get_logger().info("==================================================")

    def load_waypoints(self, path):
        if not os.path.exists(path):
            self.get_logger().error(f"Waypoint file not found: {path}")
            return np.array([])
        pts = []
        with open(path) as f:
            reader = csv.reader(f)
            next(reader, None)
            for r in reader:
                if len(r) >= 2:
                    pts.append([float(r[0]), float(r[1])])
        self.get_logger().info(f"Loaded {len(pts)} waypoints from {path}")
        return np.array(pts)

    def odom_callback(self, msg: Odometry):
        with self.lock:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.map_frame, self.base_frame, rclpy.time.Time())
                self.car_x = tf.transform.translation.x
                self.car_y = tf.transform.translation.y
                q = tf.transform.rotation
                siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
                cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                self.car_yaw = math.atan2(siny_cosp, cosy_cosp)
            except Exception:
                self.car_x = msg.pose.pose.position.x
                self.car_y = msg.pose.pose.position.y
                q = msg.pose.pose.orientation
                siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
                cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                self.car_yaw = math.atan2(siny_cosp, cosy_cosp)

            self.odom_received = True

    def scan_callback(self, msg: LaserScan):
        with self.lock:
            if not self.odom_received or len(self.waypoints) == 0:
                return
            curr_x, curr_y, curr_yaw = self.car_x, self.car_y, self.car_yaw

        cte, nearest_idx = self.calculate_cross_track_error(curr_x, curr_y)
        expert_speed, expert_steer = self.calculate_pure_pursuit(curr_x, curr_y, curr_yaw, nearest_idx)
        preprocessed_scan = self.preprocess_scan(msg)

        ai_speed, ai_steer = 0.0, 0.0
        if self.model is not None:
            ai_speed, ai_steer = self.run_model_inference(preprocessed_scan)

        is_override = (self.model is None) or (cte >= self.cte_threshold)

        if is_override:
            self.publish_drive(expert_speed, expert_steer)
            self.get_logger().warn(
                f"[REAL DAgger - OVERRIDE] CTE: {cte:.3f}m >= {self.cte_threshold}m | Expert Steer: {math.degrees(expert_steer):.1f}°",
                throttle_duration_sec=1.0
            )
        else:
            self.publish_drive(ai_speed, ai_steer)
            self.get_logger().info(
                f"[REAL DAgger - AI] CTE: {cte:.3f}m | AI Speed: {ai_speed:.2f} m/s, Steer: {math.degrees(ai_steer):.1f}°",
                throttle_duration_sec=2.0
            )

        with self.lock:
            row = list(preprocessed_scan) + [expert_speed, expert_steer]
            self.buffer.append(row)

            if len(self.buffer) >= self.buffer_size:
                buffer_to_save = list(self.buffer)
                self.buffer.clear()
                threading.Thread(target=self._flush_buffer, args=(buffer_to_save,), daemon=True).start()

    def calculate_cross_track_error(self, car_x, car_y):
        num_pts = len(self.waypoints)
        search_len = min(50, num_pts)
        indices = [(self.last_idx + i) % num_pts for i in range(search_len)]
        search_points = self.waypoints[indices]

        dists = np.linalg.norm(search_points - np.array([car_x, car_y]), axis=1)
        min_local_idx = np.argmin(dists)
        nearest_idx = indices[min_local_idx]
        self.last_idx = nearest_idx

        A = self.waypoints[nearest_idx]
        B = self.waypoints[(nearest_idx + 1) % num_pts]
        P = np.array([car_x, car_y])

        AB = B - A
        AP = P - A
        ab_len_sq = np.sum(AB**2)

        if ab_len_sq < 1e-6:
            cte = np.linalg.norm(AP)
        else:
            t = np.clip(np.dot(AP, AB) / ab_len_sq, 0.0, 1.0)
            closest_pt = A + t * AB
            cte = np.linalg.norm(P - closest_pt)

        return cte, nearest_idx

    def calculate_pure_pursuit(self, car_x, car_y, car_yaw, nearest_idx):
        num_pts = len(self.waypoints)
        lookahead_idx = nearest_idx

        while True:
            lookahead_idx = (lookahead_idx + 1) % num_pts
            dist = math.hypot(car_x - self.waypoints[lookahead_idx][0], car_y - self.waypoints[lookahead_idx][1])
            if dist >= self.lookahead_dist or lookahead_idx == nearest_idx:
                target_pt = self.waypoints[lookahead_idx]
                break

        dx = target_pt[0] - car_x
        dy = target_pt[1] - car_y
        local_y = -dx * math.sin(car_yaw) + dy * math.cos(car_yaw)

        steering_angle = math.atan((2.0 * self.wheelbase * local_y) / (self.lookahead_dist**2))
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)

        speed = self.expert_speed
        if abs(steering_angle) > 0.25:
            speed *= 0.7

        return speed, steering_angle

    def preprocess_scan(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment
        crop_limit = math.radians(60.0)

        angles = np.arange(len(ranges)) * angle_increment + angle_min
        mask = (angles >= -crop_limit) & (angles <= crop_limit)

        if not np.any(mask):
            return np.ones(self.target_beams, dtype=np.float32) * self.max_range

        valid_ranges = ranges[mask]
        valid_angles = angles[mask]
        valid_ranges = np.where(np.isnan(valid_ranges) | np.isinf(valid_ranges), self.max_range, valid_ranges)
        valid_ranges = np.clip(valid_ranges, 0.0, self.max_range)

        target_angles = np.linspace(-crop_limit, crop_limit, self.target_beams)
        return np.interp(target_angles, valid_angles, valid_ranges)

    def run_model_inference(self, preprocessed_scan):
        if not _HAS_TORCH or self.model is None:
            return 0.0, 0.0

        with torch.no_grad():
            norm_scan = preprocessed_scan / 10.0
            tensor_input = torch.tensor(norm_scan, dtype=torch.float32).unsqueeze(0).to(self.device)
            output = self.model(tensor_input).cpu().squeeze(0).numpy()

        if self.target_mean is not None and self.target_std is not None:
            output = output * self.target_std + self.target_mean

        speed = float(np.clip(output[0], 0.0, self.ai_speed))
        steering_angle = float(np.clip(output[1], -self.max_steer, self.max_steer))
        return speed, steering_angle

    def publish_drive(self, speed, steering_angle):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)

    def _write_header(self):
        with open(self.dataset_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [f'lidar_{i}' for i in range(self.target_beams)] + ['speed', 'steering_angle']
            writer.writerow(header)
        self.get_logger().info(f"Created DAgger Real Vehicle dataset at: {self.dataset_path}")

    def _flush_buffer(self, data_list):
        try:
            with open(self.dataset_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data_list)

            self.total_saved_samples += len(data_list)
            self.get_logger().info(
                f"[REAL DAgger] Saved +{len(data_list)} expert recovery samples. Total in dataset: {self.total_saved_samples}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to flush DAgger buffer: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DaggerInferenceRealPytorchNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Shutting down DAgger real vehicle node...")
        node.publish_drive(0.0, 0.0)
        with node.lock:
            remaining = list(node.buffer)
            node.buffer.clear()
        if remaining:
            node._flush_buffer(remaining)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
