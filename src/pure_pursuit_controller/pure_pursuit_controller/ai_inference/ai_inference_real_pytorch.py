#!/usr/bin/env python3
"""
ai_inference_real_pytorch.py
────────────────────────────
ROS 2 Node chạy Suy luận mô hình AI tự lái thuần túy (Autonomous Only) trên XE THẬT Jetson bằng PYTORCH.
Sử dụng file trọng số PyTorch (.pth) trực tiếp.

Do onnxruntime trên Jetson thường gặp lỗi phân bổ nhân CPU (vector assertion error),
node này sử dụng PyTorch có sẵn trên Jetson để chạy suy luận ổn định tuyệt đối.
"""

import os
import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# --- Định nghĩa kiến trúc mô hình — PHẢI khớp 100% với train.py ---
# train.py dùng Dropout(0.1) xen giữa các Linear layer, làm index trong
# nn.Sequential thay đổi so với bản không có Dropout -> state_dict key mismatch.
class DAggerMLP(nn.Module):
    def __init__(self, input_dim=60, output_dim=2, dropout=0.1):
        super(DAggerMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),  # network.0
            nn.ReLU(),                   # network.1
            nn.Dropout(dropout),         # network.2
            nn.Linear(128, 64),          # network.3
            nn.ReLU(),                   # network.4
            nn.Dropout(dropout),         # network.5
            nn.Linear(64, 32),           # network.6
            nn.ReLU(),                   # network.7
            nn.Linear(32, output_dim)    # network.8
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


class AiInferenceRealPytorchNode(Node):
    def __init__(self):
        super().__init__('ai_inference_real_pytorch_node')

        # --- Đường dẫn mặc định trỏ tới thư mục models/ ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(current_dir, '..', 'models', 'model_sim_5.pth')
        default_model_path = resolve_model_path(default_model_path)

        # --- 1. Parameters ---
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('target_beams', 60)
        self.declare_parameter('ai_speed', 1.0)           # Safe speed limit (m/s)
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('max_steering_angle', 0.35) # Max steering angle (rad)

        self.model_path = resolve_model_path(self.get_parameter('model_path').value)
        self.target_beams = self.get_parameter('target_beams').value
        self.ai_speed = self.get_parameter('ai_speed').value
        self.max_range = self.get_parameter('max_range').value
        self.max_steer = self.get_parameter('max_steering_angle').value

        # --- 2. Load PyTorch Model ---
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
                self.get_logger().error(f"Error loading normalization stats: {e}")
        else:
            self.get_logger().error(f"Normalization stats file not found at {norm_path}.")

        if _HAS_TORCH:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.get_logger().info("Using GPU (CUDA) for PyTorch inference.")
            else:
                self.device = torch.device('cpu')
                torch.set_num_threads(1)
                self.get_logger().info("CUDA not available. Using CPU for PyTorch inference.")
            
            if self.target_mean is None or self.target_std is None:
                self.get_logger().error("Model initialization aborted: normalization stats (_norm.json) missing.")
                self.model = None
            elif os.path.exists(self.model_path) and self.model_path.endswith('.pth'):
                try:
                    raw_model = DAggerMLP(input_dim=self.target_beams, output_dim=2, dropout=0.1).to(self.device)
                    raw_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                    raw_model.eval()
                    
                    self.model = torch.jit.script(raw_model)
                    self.get_logger().info(f"Successfully loaded and JIT-compiled PyTorch model from {self.model_path}")
                except Exception as e:
                    self.get_logger().error(f"Failed to load PyTorch model weights: {e}")
            else:
                self.get_logger().error(f"PyTorch Model file not found at {self.model_path}!")
        else:
            self.get_logger().error("PyTorch is not installed in this environment.")

        # --- 3. Pub/Sub ---
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.get_logger().info("=========================================")
        self.get_logger().info(" AI PYTORCH INFERENCE (REAL VEHICLE) READY")
        self.get_logger().info(f" Model Path: {self.model_path}")
        self.get_logger().info(f" Safe Speed Limit: {self.ai_speed} m/s")
        self.get_logger().info(f" Max Steer Limit: {self.max_steer} rad")
        self.get_logger().info("=========================================")

    def scan_callback(self, msg: LaserScan):
        if self.model is None:
            self.publish_drive(0.0, 0.0)
            self.get_logger().warn("PyTorch model is not loaded. Car stopped.", throttle_duration_sec=2.0)
            return

        preprocessed_scan = self.preprocess_scan(msg)
        ai_speed, ai_steer = self.run_model_inference(preprocessed_scan)

        self.publish_drive(ai_speed, ai_steer)
        self.get_logger().info(f"[REAL - PYTORCH] Speed: {ai_speed:.2f} m/s | Steer: {math.degrees(ai_steer):.1f}°", throttle_duration_sec=1.0)

    def preprocess_scan(self, msg: LaserScan):
        """Crop scan to [-60, 60] degrees and resample to 60 beams"""
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
        """Inference steering and speed from PyTorch Model"""
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
        """Publish lệnh tới mạch điều khiển VESC (/drive)"""
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AiInferenceRealPytorchNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Shutting down AI real vehicle PyTorch inference node.")
        node.publish_drive(0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
