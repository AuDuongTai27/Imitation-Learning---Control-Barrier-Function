#!/usr/bin/env python3
"""
ai_safe_controller_jetson.py
────────────────────────────
[NODE CHẠY TRÊN BO MẠCH JETSON XE THẬT]
Node AI Inference & Bộ Điều Khiển An Toàn Phân Tán (Siêu nhẹ cho Jetson CPU/GPU).

Vai trò:
  1. Suy luận AI bẻ lái từ 60 tia LiDAR qua PyTorch JIT (.pth).
  2. Đọc đề xuất bẻ lái Chuyên gia RRT* từ Laptop gửi sang qua topic trung gian `/rrt_expert_drive`.
  3. So sánh chênh lệch góc lái `steer_diff = abs(ai_steer - rrt_steer)`:
     - Nếu AI lái mượt (`steer_diff <= override_threshold`) ➔ Phát lệnh AI lên `/drive`, gửi status `1` lên `/dagger_status`.
     - Nếu RRT* cứu nét (`steer_diff > override_threshold`)  ➔ Phát lệnh RRT* lên `/drive`, gửi status `0` lên `/dagger_status` (Kích hoạt Laptop ghi DAgger CSV).

Subscribe:
  /scan             (sensor_msgs/LaserScan)                 — Cảm biến LiDAR thật trên Jetson
  /rrt_expert_drive (ackermann_msgs/AckermannDriveStamped)  — Lệnh RRT* từ Laptop truyền qua wifi

Publish:
  /drive            (ackermann_msgs/AckermannDriveStamped)  — Phát đến mạch VESC cho xe chạy
  /dagger_status    (std_msgs/Int32)                         — Báo tín hiệu về cho Laptop (0: RRT, 1: AI)
"""

import os
import math
import time
import json
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Int32

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ─────────────────────────────────────────────────────────────────────────────
#  1. KIẾN TRÚC MODEL AI PYTORCH — PHẢI KHỚP 100% VỚI TRAIN.PY
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
#  2. NODE CHÍNH (JETSON CONTROLLER)
# ─────────────────────────────────────────────────────────────────────────────

class AiSafeControllerJetsonNode(Node):
    def __init__(self):
        super().__init__('ai_safe_controller_jetson_node')

        current_dir = os.path.dirname(os.path.abspath(__file__))

        # ── PARAMETERS ───────────────────────────────────────────
        self.declare_parameter('model_path', os.path.join(current_dir, 'combine_1.pth'))
        self.declare_parameter('target_beams', 60)
        self.declare_parameter('ai_speed', 1.0)
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('max_steering_ai', 0.35)

        self.declare_parameter('override_threshold', 0.15)  # Ngưỡng bẻ lái cứu nét (rad)
        self.declare_parameter('override_hold_secs', 1.0)  # Thời gian RRT* giữ lái sau khi cứu (giây)

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('expert_drive_topic', '/rrt_expert_drive')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('status_topic', '/dagger_status')

        # Đọc parameters
        self.model_path         = self.get_parameter('model_path').value
        self.target_beams       = self.get_parameter('target_beams').value
        self.ai_speed           = self.get_parameter('ai_speed').value
        self.max_range          = self.get_parameter('max_range').value
        self.max_steer_ai       = self.get_parameter('max_steering_ai').value

        self.override_threshold  = self.get_parameter('override_threshold').value
        self.override_hold_secs  = self.get_parameter('override_hold_secs').value

        self.scan_topic         = self.get_parameter('scan_topic').value
        self.expert_drive_topic = self.get_parameter('expert_drive_topic').value
        self.drive_topic        = self.get_parameter('drive_topic').value
        self.status_topic       = self.get_parameter('status_topic').value

        # ── STATE ────────────────────────────────────────────────
        self.lock = threading.Lock()

        self.latest_expert_drive = None
        self.latest_expert_time  = 0.0

        self.override_active     = False
        self.override_until      = 0.0

        # ── LOAD PYTORCH MODEL ───────────────────────────────────
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
                self.get_logger().info(f"Loaded Norm Stats: mean={self.target_mean}, std={self.target_std}")
            except Exception as e:
                self.get_logger().warn(f"Cannot load norm stats: {e}")

        if _HAS_TORCH:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if os.path.exists(self.model_path) and self.model_path.endswith('.pth'):
                try:
                    raw_model = DAggerMLP(input_dim=self.target_beams, output_dim=2, dropout=0.1).to(self.device)
                    raw_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                    raw_model.eval()
                    self.model = torch.jit.script(raw_model)
                    self.get_logger().info(f"AI PyTorch Model JIT loaded from: {self.model_path}")
                except Exception as e:
                    self.get_logger().error(f"Failed to load PyTorch model: {e}")
            else:
                self.get_logger().error(f"Model file not found: {self.model_path}")

        # ── PUB/SUB ──────────────────────────────────────────────
        self.drive_pub  = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.status_pub = self.create_publisher(Int32, self.status_topic, 10)

        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.create_subscription(AckermannDriveStamped, self.expert_drive_topic, self.expert_drive_callback, 10)

        self._log_startup()

    def expert_drive_callback(self, msg: AckermannDriveStamped):
        with self.lock:
            self.latest_expert_drive = msg
            self.latest_expert_time  = time.monotonic()

    def scan_callback(self, msg: LaserScan):
        now = time.monotonic()

        # 1. AI Inference
        ai_steer, ai_spd = self._run_ai(msg)

        # 2. Lấy lệnh RRT* từ Laptop
        with self.lock:
            expert_msg  = self.latest_expert_drive
            expert_age  = now - self.latest_expert_time

        expert_valid = (expert_msg is not None) and (expert_age < 0.5)

        if expert_valid:
            expert_steer = expert_msg.drive.steering_angle
            expert_speed = expert_msg.drive.speed
        else:
            expert_steer = 0.0
            expert_speed = 0.0

        # 3. So sánh chênh lệch góc lái & Quyết định Override
        if expert_valid:
            steer_diff = abs(ai_steer - expert_steer)
            if steer_diff > self.override_threshold:
                with self.lock:
                    self.override_active = True
                    self.override_until  = now + self.override_hold_secs

        with self.lock:
            if self.override_active and now >= self.override_until:
                self.override_active = False
            is_override = self.override_active

        # 4. Lựa chọn Lệnh Lái & Phát Status về cho Laptop
        if is_override and expert_valid:
            active_mode = 'RRT_EXPERT'
            chosen_steer = expert_steer
            chosen_speed = expert_speed
            dagger_status_code = 0  # 0: RRT* đang cứu nét -> Laptop lưu DAgger CSV
        else:
            active_mode = 'AI'
            chosen_steer = ai_steer
            chosen_speed = ai_spd
            dagger_status_code = 1  # 1: AI đang tự lái -> Laptop dừng ghi

        # Publish lệnh lên VESC /drive
        self._publish_drive(chosen_steer, chosen_speed)

        # Publish status về Laptop
        status_msg = Int32()
        status_msg.data = dagger_status_code
        self.status_pub.publish(status_msg)

        # In log đối chiếu
        steer_diff = abs(ai_steer - expert_steer) if expert_valid else 0.0
        self.get_logger().info(
            f"[{active_mode:10s}] "
            f"AI: [steer={math.degrees(ai_steer):+5.1f}°, spd={ai_spd:.2f}m/s] | "
            f"RRT* Laptop: [steer={math.degrees(expert_steer):+5.1f}°, spd={expert_speed:.2f}m/s] | "
            f"Diff: {math.degrees(steer_diff):.1f}°",
            throttle_duration_sec=0.5
        )

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

    def _publish_drive(self, steer: float, speed: float):
        msg = AckermannDriveStamped()
        msg.header.stamp        = self.get_clock().now().to_msg()
        msg.header.frame_id     = 'laser'
        msg.drive.steering_angle = float(steer)
        msg.drive.speed          = float(speed)
        self.drive_pub.publish(msg)

    def _log_startup(self):
        self.get_logger().info("=" * 60)
        self.get_logger().info(" 🚀 AI SAFE CONTROLLER (JETSON BOARD) STARTED")
        self.get_logger().info(f"  Model Path  : {self.model_path}")
        self.get_logger().info(f"  AI Speed    : {self.ai_speed} m/s")
        self.get_logger().info(f"  RRT* Relay  : {self.expert_drive_topic}")
        self.get_logger().info(f"  Status Topic: {self.status_topic}")
        self.get_logger().info(f"  Threshold   : {math.degrees(self.override_threshold):.1f}° ({self.override_threshold:.3f} rad)")
        self.get_logger().info("=" * 60)


def main(args=None):
    rclpy.init(args=args)
    node = AiSafeControllerJetsonNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Shutting down Jetson AI Safe Controller — stopping car.")
        node._publish_drive(0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
