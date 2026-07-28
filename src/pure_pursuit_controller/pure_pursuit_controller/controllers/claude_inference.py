#!/usr/bin/env python3
"""
ai_inference_real_pytorch.py
────────────────────────────
ROS 2 Node chạy Suy luận mô hình AI tự lái thuần túy (Autonomous Only) trên XE THẬT Jetson bằng PYTORCH.
Sử dụng file trọng số PyTorch (.pth) trực tiếp — model được train bằng train.py MỚI
(có Dropout + target chuẩn hóa z-score).

Do onnxruntime trên Jetson thường gặp lỗi phân bổ nhân CPU (vector assertion error),
node này dùng PyTorch có sẵn trên Jetson để chạy suy luận ổn định.

QUAN TRỌNG so với bản cũ:
  1. Kiến trúc DAggerMLP có thêm Dropout để khớp đúng với train.py mới — nếu thiếu,
     load_state_dict() sẽ lệch key (network.2/4/6 vs network.3/6/8) và báo lỗi hoặc
     load sai trọng số.
  2. Output của model là dạng ĐÃ CHUẨN HÓA (z-score) — bắt buộc phải denormalize
     bằng target_mean/target_std đọc từ file *_norm.json (được train.py tự sinh ra
     cạnh file .pth) trước khi dùng làm (linear_v, angular_z) thật.
"""

import os
import json
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


# --- Định nghĩa kiến trúc mô hình — PHẢI khớp 100% với train.py đã tạo ra file .pth ---
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


class AiInferenceRealPytorchNode(Node):
    def __init__(self):
        super().__init__('ai_inference_real_pytorch_node')

        # --- Đường dẫn mặc định nằm cùng cấp với script ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(current_dir, 'dagger_model_sim_4.pth')

        # --- 1. Parameters ---
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('norm_path', '')  # rỗng -> tự suy ra từ model_path
        self.declare_parameter('target_beams', 60)
        self.declare_parameter('ai_speed', 1.0)            # ⚠️ Tốc độ chạy thử xe thật an toàn (1.0 m/s)
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('max_steering_angle', 0.35)  # Góc lái vật lý xe thật (rad)

        self.model_path = self.get_parameter('model_path').value
        norm_path_param = self.get_parameter('norm_path').value
        self.target_beams = self.get_parameter('target_beams').value
        self.ai_speed = self.get_parameter('ai_speed').value
        self.max_range = self.get_parameter('max_range').value
        self.max_steer = self.get_parameter('max_steering_angle').value

        self.norm_path = norm_path_param if norm_path_param else (
            os.path.splitext(self.model_path)[0] + '_norm.json'
        )

        # --- 2. Load normalization stats (bắt buộc, không có thì KHÔNG chạy) ---
        self.target_mean = None
        self.target_std = None
        if os.path.exists(self.norm_path):
            try:
                with open(self.norm_path, 'r') as f:
                    norm_stats = json.load(f)
                self.target_mean = np.array(norm_stats['target_mean'], dtype=np.float32)
                self.target_std = np.array(norm_stats['target_std'], dtype=np.float32)
                # Ưu tiên lidar_max_range lấy từ file norm, override lại param nếu có
                self.max_range = float(norm_stats.get('lidar_max_range', self.max_range))
                self.get_logger().info(
                    f"Loaded normalization stats từ {self.norm_path} "
                    f"(target_mean={self.target_mean.tolist()}, target_std={self.target_std.tolist()}, "
                    f"lidar_max_range={self.max_range})"
                )
            except Exception as e:
                self.get_logger().error(f"Failed to load normalization stats: {e}")
        else:
            self.get_logger().error(
                f"KHÔNG tìm thấy file normalization stats tại {self.norm_path}. "
                f"Model sẽ trả ra output SAI (dạng z-score, không phải m/s, rad thật) nếu thiếu file này. "
                f"Node sẽ không load model để tránh chạy sai."
            )

        # --- 3. Load PyTorch Model (chỉ load nếu đã có norm stats hợp lệ) ---
        self.model = None
        if _HAS_TORCH and self.target_mean is not None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.get_logger().info("Using GPU (CUDA) for PyTorch inference.")
            else:
                self.device = torch.device('cpu')
                torch.set_num_threads(1)
                self.get_logger().info("CUDA not available. Using CPU for PyTorch inference.")

            if os.path.exists(self.model_path) and self.model_path.endswith('.pth'):
                try:
                    raw_model = DAggerMLP(input_dim=self.target_beams, output_dim=2).to(self.device)
                    raw_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                    raw_model.eval()

                    # 🚀 TỐI ƯU HÓA: Dùng TorchScript JIT Compiler để biên dịch model
                    self.model = torch.jit.script(raw_model)
                    self.model.eval()
                    self.get_logger().info(f"Successfully loaded and JIT-compiled PyTorch model from {self.model_path}")
                except Exception as e:
                    self.get_logger().error(f"Failed to load PyTorch model weights: {e}")
            else:
                self.get_logger().error(f"PyTorch Model file not found at {self.model_path}!")
        elif not _HAS_TORCH:
            self.get_logger().error("PyTorch is not installed in this environment! Cannot run PyTorch inference.")

        # --- 4. Pub/Sub ---
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.get_logger().info("=========================================")
        self.get_logger().info(" AI PYTORCH INFERENCE (REAL VEHICLE) READY")
        self.get_logger().info(f" Model Path: {self.model_path}")
        self.get_logger().info(f" Norm Path:  {self.norm_path}")
        self.get_logger().info(f" Safe Speed Limit: {self.ai_speed} m/s")
        self.get_logger().info(f" Max Steer Limit: {self.max_steer} rad")
        self.get_logger().info("=========================================")

    def scan_callback(self, msg: LaserScan):
        """Xử lý dữ liệu scan, chạy suy luận PyTorch và publish lệnh điều khiển"""
        if self.model is None:
            self.publish_drive(0.0, 0.0)
            self.get_logger().warn("PyTorch model is not loaded. Car stopped.", throttle_duration_sec=2.0)
            return

        # 1. Tiền xử lý dữ liệu scan (Crop góc quét & downsample)
        preprocessed_scan = self.preprocess_scan(msg)

        # 2. Suy luận qua mô hình PyTorch
        ai_speed, ai_steer = self.run_model_inference(preprocessed_scan)

        # 3. Điều khiển xe chạy tự động
        self.publish_drive(ai_speed, ai_steer)
        self.get_logger().info(f"[REAL - PYTORCH] Speed: {ai_speed:.2f} m/s | Steer: {math.degrees(ai_steer):.1f}°", throttle_duration_sec=1.0)

    def preprocess_scan(self, msg: LaserScan):
        """Crop góc quét về [-60, 60] độ và downsample về 60 beams"""
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
        """Dự đoán lệnh lái Ackermann qua PyTorch Model, có denormalize output"""
        if not _HAS_TORCH or self.model is None:
            return 0.0, 0.0

        with torch.no_grad():
            # Chuẩn hóa input giống lúc train: chia cho lidar_max_range
            norm_scan = preprocessed_scan / self.max_range
            tensor_input = torch.tensor(norm_scan, dtype=torch.float32).unsqueeze(0).to(self.device)
            output_norm = self.model(tensor_input).cpu().squeeze(0).numpy()

        # Denormalize: output model là z-score -> quy đổi lại (linear_v, angular_z) thật
        output_real = output_norm * self.target_std + self.target_mean

        # Output: [speed, steering_angle]
        speed = float(np.clip(output_real[0], 0.0, self.ai_speed))
        steering_angle = float(np.clip(output_real[1], -self.max_steer, self.max_steer))
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