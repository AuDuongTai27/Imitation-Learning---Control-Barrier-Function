#!/usr/bin/env python3
"""
ai_inference_real_pytorch.py
────────────────────────────
ROS 2 Node chạy Suy luận mô hình AI tự lái trên XE THẬT Jetson bằng PYTORCH,
hỗ trợ tích hợp bộ lọc an toàn CBF-QP (Control Barrier Function) và Bộ lọc làm mượt chống giật cục.

Được trang bị 2 chế độ linh hoạt:
  1. Chế độ 1-Node Tích hợp Lõi CBF + Làm Mượt (Tối ưu nhất cho Jetson - Chỉ 1 Terminal):
     -p enable_cbf:=true -p enable_smoothing:=true -> Tự động suy luận, làm mượt và lọc CBF phát thẳng ra /drive
  2. Chế độ 2-Node ROS 2 (Thô / Cascaded):
     -p drive_topic:=/drive_raw -> Phát lệnh thô AI ra /drive_raw để cbf_smooth_safety_filter_node.py xử lý
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

# Import bộ lọc an toàn CBF
try:
    from pure_pursuit_controller.cbf_core import CBFQPSafetyFilter
    _HAS_CBF = True
except ImportError:
    try:
        from cbf_core import CBFQPSafetyFilter
        _HAS_CBF = True
    except ImportError:
        _HAS_CBF = False


# --- Định nghĩa kiến trúc mô hình — PHẢI khớp 100% với train.py ---
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

        # --- Đường dẫn mặc định ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(current_dir, '..', 'models', 'model_sim_5.pth')
        default_model_path = resolve_model_path(default_model_path)

        # --- 1. Parameters ---
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('target_beams', 60)
        self.declare_parameter('ai_speed', 1.5)            # Tốc độ tối đa giới hạn
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('max_steering_angle', 0.35)  # Góc lái vật lý xe thật (rad ~ 20 deg)
        self.declare_parameter('drive_topic', '/drive_raw')  # Mặc định phát ra /drive_raw khi ghép CBF
        self.declare_parameter('scan_topic', '/scan')       # Topic LiDAR xe thật
        self.declare_parameter('speed_scale', 1.0)         # Hệ số nhân tốc độ AI
        self.declare_parameter('fixed_speed', 0.0)         # Ép tốc độ chạy cố định nếu > 0

        # --- Tham số cho bộ lọc CBF & Làm Mượt tích hợp ---
        self.declare_parameter('enable_cbf', False)         # Bật/Tắt CBF trực tiếp trong node
        self.declare_parameter('enable_smoothing', True)    # Bật/Tắt làm mượt chống giật
        self.declare_parameter('alpha_steer', 0.30)         # Hệ số mượt tay lái
        self.declare_parameter('alpha_speed', 0.35)         # Hệ số mượt tốc độ
        self.declare_parameter('d_min', 0.25)               # Khoảng cách an toàn tối thiểu (m)
        self.declare_parameter('cbf_gamma', 3.5)            # Hệ số CBF gain
        self.declare_parameter('a_max_brake', 2.61)         # Gia tốc phanh xe thật
        self.declare_parameter('wheelbase', 0.39)           # Chiều dài cơ sở xe thật

        self.model_path = resolve_model_path(self.get_parameter('model_path').value)
        self.target_beams = self.get_parameter('target_beams').value
        self.ai_speed = self.get_parameter('ai_speed').value
        self.max_range = self.get_parameter('max_range').value
        self.max_steer = self.get_parameter('max_steering_angle').value
        self.drive_topic = self.get_parameter('drive_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.speed_scale = self.get_parameter('speed_scale').value
        self.fixed_speed = self.get_parameter('fixed_speed').value

        self.enable_cbf = self.get_parameter('enable_cbf').value
        self.enable_smoothing = self.get_parameter('enable_smoothing').value
        self.alpha_steer = self.get_parameter('alpha_steer').value
        self.alpha_speed = self.get_parameter('alpha_speed').value
        self.d_min = self.get_parameter('d_min').value
        self.cbf_gamma = self.get_parameter('cbf_gamma').value
        self.a_max_brake = self.get_parameter('a_max_brake').value
        self.wheelbase = self.get_parameter('wheelbase').value

        # Trạng thái mượt
        self.prev_smooth_speed = 0.0
        self.prev_smooth_steer = 0.0

        # Trực tiếp mở CBF lõi nếu được bật
        self.cbf_filter = None
        if self.enable_cbf:
            if _HAS_CBF:
                self.cbf_filter = CBFQPSafetyFilter(
                    d_min=self.d_min,
                    gamma=self.cbf_gamma,
                    v_max=self.ai_speed,
                    steer_max=self.max_steer,
                    a_max_brake=self.a_max_brake,
                    wheelbase=self.wheelbase
                )
                self.get_logger().info(f"[CBF ENABLED] Safety Shield Active (d_min={self.d_min}m, gamma={self.cbf_gamma})")
                self.drive_topic = '/drive'
            else:
                self.get_logger().error("CBFQPSafetyFilter module not found! Cannot enable built-in CBF.")

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
                self.get_logger().info(f"Loaded norm stats: mean={self.target_mean}, std={self.target_std}")
            except Exception as e:
                self.get_logger().warn(f"Failed to load norm stats: {e}")

        if _HAS_TORCH:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.get_logger().info("Using GPU (CUDA) for PyTorch inference.")
            else:
                self.device = torch.device('cpu')
                torch.set_num_threads(1)
                self.get_logger().info("CUDA not available. Using CPU for PyTorch inference.")
            
            if os.path.exists(self.model_path) and self.model_path.endswith('.pth'):
                try:
                    loaded = torch.load(self.model_path, map_location=self.device)
                    raw_model = DAggerMLP(input_dim=self.target_beams, output_dim=2, dropout=0.1).to(self.device)
                    
                    if isinstance(loaded, nn.Module):
                        raw_model = loaded
                    elif isinstance(loaded, dict):
                        if 'model_state_dict' in loaded:
                            raw_model.load_state_dict(loaded['model_state_dict'])
                        elif 'state_dict' in loaded:
                            raw_model.load_state_dict(loaded['state_dict'])
                        else:
                            raw_model.load_state_dict(loaded)
                    
                    raw_model.eval()
                    try:
                        self.model = torch.jit.script(raw_model)
                    except Exception:
                        self.model = raw_model

                    self.get_logger().info(f"Successfully loaded PyTorch model from {self.model_path}")
                except Exception as e:
                    self.get_logger().error(f"Failed to load PyTorch model: {e}")
            else:
                self.get_logger().error(f"PyTorch Model file not found at {self.model_path}!")
        else:
            self.get_logger().error("PyTorch is not installed in this environment!")

        # --- 3. Pub/Sub ---
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.raw_drive_pub = self.create_publisher(AckermannDriveStamped, '/drive_raw', 10)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)

        self.get_logger().info("=========================================")
        self.get_logger().info(" AI PYTORCH INFERENCE (REAL VEHICLE) READY")
        self.get_logger().info(f" Model Path  : {self.model_path}")
        self.get_logger().info(f" Drive Topic : {self.drive_topic}")
        self.get_logger().info(f" Raw Topic   : /drive_raw (Live Plotting)")
        self.get_logger().info(f" Scan Topic  : {self.scan_topic}")
        self.get_logger().info(f" Built-in CBF: {self.enable_cbf}")
        self.get_logger().info(f" Smoothing   : {self.enable_smoothing}")
        self.get_logger().info("=========================================")

    def scan_callback(self, msg: LaserScan):
        """Xử lý scan, suy luận AI, làm mượt EMA và lọc an toàn bằng CBF"""
        if self.model is None:
            self.publish_drive(0.0, 0.0)
            self.get_logger().warn("PyTorch model not loaded. Car stopped.", throttle_duration_sec=2.0)
            return

        # 1. Tiền xử lý scan LiDAR
        preprocessed_scan = self.preprocess_scan(msg)

        # 2. Suy luận lệnh lái AI thô [v_ai, steer_ai]
        ai_speed, ai_steer = self.run_model_inference(preprocessed_scan)
        self.publish_raw_drive(ai_speed, ai_steer)

        # 3. Bộ lọc làm mượt EMA chống giật cục
        if self.enable_smoothing:
            smooth_speed = (1.0 - self.alpha_speed) * self.prev_smooth_speed + self.alpha_speed * ai_speed
            smooth_steer = (1.0 - self.alpha_steer) * self.prev_smooth_steer + self.alpha_steer * ai_steer
            self.prev_smooth_speed = smooth_speed
            self.prev_smooth_steer = smooth_steer
        else:
            smooth_speed = ai_speed
            smooth_steer = ai_steer

        # 4. Lọc qua CBF Safety Shield nếu được bật
        final_speed, final_steer = smooth_speed, smooth_steer
        if self.cbf_filter is not None:
            ranges = np.array(msg.ranges, dtype=np.float32)
            angles = np.arange(len(ranges), dtype=np.float32) * msg.angle_increment + msg.angle_min
            
            u_nom = np.array([smooth_speed, smooth_steer], dtype=np.float32)
            u_safe = self.cbf_filter.filter(u_nom, ranges, angles)
            
            final_speed, final_steer = float(u_safe[0]), float(u_safe[1])

            if abs(final_speed - ai_speed) > 0.05 or abs(final_steer - ai_steer) > 0.05:
                self.get_logger().warn(
                    f"[CBF REAL SHIELD] AI: v={ai_speed:.2f}, steer={math.degrees(ai_steer):.1f}° | "
                    f"Safe: v={final_speed:.2f}, steer={math.degrees(final_steer):.1f}°",
                    throttle_duration_sec=0.5
                )

        # 5. Điều khiển xe
        self.publish_drive(final_speed, final_steer)
        self.get_logger().info(
            f"[REAL VEHICLE] Speed: {final_speed:.2f} m/s | Steer: {math.degrees(final_steer):.1f}°",
            throttle_duration_sec=1.0
        )

    def publish_raw_drive(self, speed, steering_angle):
        """Publish lệnh thô AI ra /drive_raw để cbf_live_plotter.py vẽ đồ thị"""
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.raw_drive_pub.publish(msg)

    def preprocess_scan(self, msg: LaserScan):
        """Crop [-60, 60] độ và downsample về target_beams"""
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
        """Dự đoán lệnh lái Ackermann qua PyTorch Model (.pth)"""
        if not _HAS_TORCH or self.model is None:
            return 0.0, 0.0

        with torch.no_grad():
            norm_scan = preprocessed_scan / 10.0
            tensor_input = torch.tensor(norm_scan, dtype=torch.float32).unsqueeze(0).to(self.device)
            output = self.model(tensor_input).cpu().squeeze(0).numpy()

        if self.target_mean is not None and self.target_std is not None:
            output = output * self.target_std + self.target_mean

        raw_speed = float(output[0])
        if self.fixed_speed > 0.0:
            speed = float(np.clip(self.fixed_speed, 0.0, self.ai_speed))
        else:
            speed = float(np.clip(raw_speed * self.speed_scale, 0.0, self.ai_speed))

        steering_angle = float(np.clip(output[1], -self.max_steer, self.max_steer))
        return speed, steering_angle

    def publish_drive(self, speed, steering_angle):
        """Publish lệnh tới mạch điều khiển VESC"""
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
