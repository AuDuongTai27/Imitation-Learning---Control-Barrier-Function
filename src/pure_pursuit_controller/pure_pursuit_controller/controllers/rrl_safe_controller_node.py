#!/usr/bin/env python3
"""
rrl_safe_controller_node.py (V2 - Updated with FOV Crop, Denormalization, and Real-time Debug Logs)
─────────────────────────────────────────────────────────────────────────────────────────
ROS 2 Deployment Node for Residual Reinforcement Learning (RRL) + CBF Safety Filter.
Supports both PyTorch (.pth) and ONNXRuntime (.onnx) models.

Pipeline:
1. Subscribes to `/scan` (LiDAR) and `/odom` (Odometry).
2. Preprocesses LiDAR: Crops frontal FOV [-60, 60] degrees and resamples to 60 normalized beams.
3. Evaluates frozen DAgger DIL Baseline Model -> a_DIL = [v_DIL, delta_DIL] (denormalized via target_mean/std).
4. Evaluates trained RRL Policy Network (PyTorch or ONNX) -> a_R = [Δv, Δδ].
5. Sums total control: a_total = a_DIL + a_R.
6. Filters via CBF-QP Safety Layer -> a_safe = [v_safe, delta_safe].
7. Publishes final control command to `/drive`.
"""

import os
import json
import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if _HAS_TORCH:
    try:
        from pure_pursuit_controller.training.train import DAggerMLP
        from pure_pursuit_controller.training.rrl.rrl_model import RRLActorCritic
    except ImportError:
        pass

try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False

from pure_pursuit_controller.cbf.cbf_core import CBFQPSafetyFilter


def resolve_model_path(path_or_name: str) -> str:
    if not path_or_name:
        return ''
    if os.path.exists(path_or_name):
        return path_or_name

    filename = os.path.basename(path_or_name)
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    search_dirs = [
        '/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/models',
        '/home/adt/f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/models',
        os.path.abspath(os.path.join(curr_dir, '..', 'models')),
        os.path.abspath(os.path.join(curr_dir, 'models'))
    ]

    for sdir in search_dirs:
        candidate = os.path.join(sdir, filename)
        if os.path.exists(candidate):
            return candidate

    return path_or_name


class RRLSafeControllerNode(Node):
    def __init__(self):
        super().__init__('rrl_safe_controller_node')

        self.get_logger().info("=========================================")
        self.get_logger().info(" RRL SAFE CONTROLLER NODE V2 LAUNCHED   ")
        self.get_logger().info("=========================================")

        self.declare_parameter('dagger_model_path', '')
        self.declare_parameter('rrl_model_path', '')
        self.declare_parameter('norm_param_path', '')
        self.declare_parameter('use_cbf', True)
        self.declare_parameter('enable_rrl', True)
        self.declare_parameter('drive_topic', '/drive')

        dagger_path = self.get_parameter('dagger_model_path').get_parameter_value().string_value
        rrl_path = self.get_parameter('rrl_model_path').get_parameter_value().string_value
        norm_path = self.get_parameter('norm_param_path').get_parameter_value().string_value
        self.use_cbf = self.get_parameter('use_cbf').get_parameter_value().bool_value
        self.enable_rrl = self.get_parameter('enable_rrl').get_parameter_value().bool_value
        drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        # Default paths fallback if not specified via ROS parameters
        if not dagger_path:
            dagger_path = 'final_combined_37500.onnx'
        dagger_path = resolve_model_path(dagger_path)

        if not rrl_path:
            rrl_path = 'rrl_ppo_model.onnx'
        rrl_path = resolve_model_path(rrl_path)

        if not norm_path:
            norm_path = 'final_combined_37500_norm.json'
        norm_path = resolve_model_path(norm_path)

        # If PyTorch is unavailable, force fallback to ONNX models if present
        if not _HAS_TORCH:
            if dagger_path.endswith('.pth') and os.path.exists(dagger_path.replace('.pth', '.onnx')):
                dagger_path = dagger_path.replace('.pth', '.onnx')
            if rrl_path.endswith('.pth') and os.path.exists(rrl_path.replace('.pth', '.onnx')):
                rrl_path = rrl_path.replace('.pth', '.onnx')

        self.get_logger().info(f"DIL Model Path: {dagger_path}")
        self.get_logger().info(f"RRL Model Path: {rrl_path}")
        self.get_logger().info(f"Norm Param Path: {norm_path}")
        self.get_logger().info(f"Enable RRL: {self.enable_rrl} | Use CBF: {self.use_cbf}")

        # Load Normalization Params
        with open(norm_path, 'r') as f:
            norm_data = json.load(f)
        self.target_beams = norm_data.get("target_beams", 60)
        self.max_range = norm_data.get("max_range", 10.0)

        if "target_mean" in norm_data and "target_std" in norm_data:
            self.target_mean = np.array(norm_data["target_mean"], dtype=np.float32)
            self.target_std = np.array(norm_data["target_std"], dtype=np.float32)
            self.get_logger().info(f"Loaded Target Mean: {self.target_mean}, Target Std: {self.target_std}")
        else:
            self.target_mean = None
            self.target_std = None

        # 1. Setup DIL Baseline Model (ONNX or PyTorch)
        self.dil_is_onnx = dagger_path.endswith('.onnx')
        if self.dil_is_onnx and _HAS_ORT:
            self.dil_ort_session = ort.InferenceSession(dagger_path)
            self.dil_input_name = self.dil_ort_session.get_inputs()[0].name
            self.get_logger().info(f"Loaded DIL Baseline from ONNX: {dagger_path}")
        elif _HAS_TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dil_model = DAggerMLP(input_dim=self.target_beams, output_dim=2).to(self.device)
            self.dil_model.load_state_dict(torch.load(dagger_path, map_location=self.device))
            self.dil_model.eval()
            self.get_logger().info(f"Loaded DIL Baseline from PyTorch: {dagger_path}")
        else:
            raise RuntimeError("Neither ONNXRuntime nor PyTorch is available for DIL model inference.")

        # 2. Setup RRL Policy Model (ONNX or PyTorch)
        self.rrl_is_onnx = rrl_path.endswith('.onnx')
        if self.rrl_is_onnx and _HAS_ORT:
            if os.path.exists(rrl_path):
                self.rrl_ort_session = ort.InferenceSession(rrl_path)
                self.rrl_input_name = self.rrl_ort_session.get_inputs()[0].name
                self.get_logger().info(f"Loaded RRL Policy from ONNX: {rrl_path}")
            else:
                self.get_logger().warn(f"RRL ONNX file not found at {rrl_path}. RRL offset will be zero.")
                self.rrl_ort_session = None
        elif _HAS_TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.rrl_model = RRLActorCritic(state_dim=self.target_beams + 6, action_dim=2).to(self.device)
            if os.path.exists(rrl_path):
                self.rrl_model.load_state_dict(torch.load(rrl_path, map_location=self.device))
                self.get_logger().info(f"Loaded RRL Policy from PyTorch: {rrl_path}")
            else:
                self.get_logger().warn(f"RRL PyTorch file not found at {rrl_path}. Running zero-init model.")
            self.rrl_model.eval()
        else:
            self.rrl_ort_session = None

        # 3. CBF Safety Filter
        self.cbf = CBFQPSafetyFilter(d_min=0.30, gamma=1.5, v_max=5.0, steer_max=0.41)

        # State Variables
        self.latest_scan = None
        self.latest_scan_angles = None
        self.v_x = 0.0
        self.omega = 0.0
        self.prev_a_R = np.zeros(2, dtype=np.float32)

        # Subscriptions & Publishers
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        # Control Loop Timer @ 30 Hz
        self.timer = self.create_timer(1.0 / 30.0, self.control_loop)

        self.get_logger().info("RRL Safe Controller Node V2 ready!")

    def odom_callback(self, msg: Odometry):
        self.v_x = float(msg.twist.twist.linear.x)
        self.omega = float(msg.twist.twist.angular.z)

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=self.max_range, posinf=self.max_range, neginf=0.0)
        self.latest_scan = ranges
        
        angles = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min
        self.latest_scan_angles = angles

    def _preprocess_scan_60_fov(self):
        """Crop LiDAR scan to [-60, 60] deg FOV and resample to 60 beams.
        Returns:
            scan_meters: unnormalized ranges in meters [0, max_range]
            scan_norm: normalized ranges [0, 1]
        """
        if self.latest_scan is None or self.latest_scan_angles is None:
            return np.ones(self.target_beams, dtype=np.float32) * self.max_range, np.ones(self.target_beams, dtype=np.float32)

        crop_limit = math.radians(60.0)
        mask = (self.latest_scan_angles >= -crop_limit) & (self.latest_scan_angles <= crop_limit)

        if not np.any(mask):
            return np.ones(self.target_beams, dtype=np.float32) * self.max_range, np.ones(self.target_beams, dtype=np.float32)

        valid_ranges = np.clip(self.latest_scan[mask], 0.0, self.max_range)
        valid_angles = self.latest_scan_angles[mask]

        target_angles = np.linspace(-crop_limit, crop_limit, self.target_beams)
        interpolated = np.interp(target_angles, valid_angles, valid_ranges)
        scan_meters = interpolated.astype(np.float32)
        scan_norm = (interpolated / self.max_range).astype(np.float32)
        return scan_meters, scan_norm

    def _infer_dil(self, scan_meters: np.ndarray, scan_norm: np.ndarray) -> np.ndarray:
        """Run DIL baseline model inference and apply denormalization if required"""
        if self.dil_is_onnx:
            if self.dil_input_name == 'lidar_raw':
                tensor_input = scan_meters.reshape(1, -1).astype(np.float32)
                outputs = self.dil_ort_session.run(None, {self.dil_input_name: tensor_input})
                raw_output = outputs[0].squeeze(0)
                # ONNX models with input_name 'lidar_raw' already output physical control commands
                return raw_output.astype(np.float32)
            else:
                tensor_input = scan_norm.reshape(1, -1).astype(np.float32)
                outputs = self.dil_ort_session.run(None, {self.dil_input_name: tensor_input})
                raw_output = outputs[0].squeeze(0)
        else:
            scan_tensor = torch.tensor(scan_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                raw_output = self.dil_model(scan_tensor).cpu().numpy().squeeze(0)

        # Denormalize target if mean/std exist in norm params
        if self.target_mean is not None and self.target_std is not None:
            raw_output = raw_output * self.target_std + self.target_mean

        return raw_output.astype(np.float32)

    def _infer_rrl(self, rrl_state: np.ndarray) -> np.ndarray:
        """Run RRL Policy model inference"""
        if self.rrl_is_onnx:
            if self.rrl_ort_session is None:
                return np.zeros(2, dtype=np.float32)
            tensor_input = rrl_state.reshape(1, -1).astype(np.float32)
            outputs = self.rrl_ort_session.run(None, {self.rrl_input_name: tensor_input})
            return outputs[0].squeeze(0)
        elif _HAS_TORCH:
            state_tensor = torch.tensor(rrl_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                a_R, _, _ = self.rrl_model.get_action(state_tensor, deterministic=True)
                return a_R.cpu().numpy().squeeze(0)
        else:
            return np.zeros(2, dtype=np.float32)

    def control_loop(self):
        if self.latest_scan is None:
            return

        # 1. Preprocess LiDAR scan to frontal 60 beams [-60, 60] deg FOV
        scan_meters, scan_norm = self._preprocess_scan_60_fov()

        # 2. DIL Baseline Inference (denormalized [v_DIL, delta_DIL])
        a_dil = self._infer_dil(scan_meters, scan_norm)

        # 3. Construct RRL State Observation (66 dimensions)
        rrl_state = np.hstack([scan_norm, [self.v_x, self.omega], a_dil, self.prev_a_R]).astype(np.float32)

        # 4. RRL Policy Inference
        if self.enable_rrl:
            raw_a_R = self._infer_rrl(rrl_state)
            # Dynamic speed adaptation bounds: allow deceleration down to -1.5 m/s in corners, and acceleration up to +2.5 m/s
            raw_a_R[0] = float(np.clip(raw_a_R[0], -1.5, 2.5))
            # EMA Low-pass Action Filtering (alpha=0.35) for smooth driving
            a_R_np = 0.35 * raw_a_R + 0.65 * self.prev_a_R
        else:
            a_R_np = np.zeros(2, dtype=np.float32)

        # 5. Total Control: a_total = a_DIL + a_R
        v_total = float(np.clip(a_dil[0] + a_R_np[0], 0.5, 7.0))
        steer_total = float(a_dil[1] + a_R_np[1])
        u_total = np.array([v_total, steer_total], dtype=np.float32)

        # 6. CBF Safety Filter Override
        if self.use_cbf and self.latest_scan_angles is not None:
            u_executed = self.cbf.filter(u_total, self.latest_scan, self.latest_scan_angles)
        else:
            u_executed = u_total

        self.prev_a_R = a_R_np

        # Debug Log Output every 0.5 sec
        self.get_logger().info(
            f"DIL: [spd={a_dil[0]:.2f}m/s, steer={math.degrees(a_dil[1]):+5.1f}°] | "
            f"RRL: [Δv={a_R_np[0]:+.2f}, Δsteer={math.degrees(a_R_np[1]):+5.1f}°] | "
            f"FINAL: [spd={u_executed[0]:.2f}m/s, steer={math.degrees(u_executed[1]):+5.1f}°]",
            throttle_duration_sec=0.5
        )

        # 7. Publish AckermannDriveStamped
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.speed = float(u_executed[0])
        drive_msg.drive.steering_angle = float(u_executed[1])
        self.pub_drive.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RRLSafeControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
