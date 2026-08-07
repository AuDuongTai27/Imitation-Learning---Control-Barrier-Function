#!/usr/bin/env python3
"""
ai_inference_real_pytorch.py
────────────────────────────
ROS 2 Node chạy Suy luận mô hình AI tự lái trên XE THẬT Jetson Nano bằng PYTORCH,
hỗ trợ tích hợp bộ lọc an toàn CBF-QP (Control Barrier Function) và Bộ lọc làm mượt.
Được tích hợp hệ thống giám sát tần số/độ trễ thời gian thực.
"""
import os
import math
import time
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

try:
    from pure_pursuit_controller.cbf_core import CBFQPSafetyFilter
    _HAS_CBF = True
except ImportError:
    try:
        from cbf_core import CBFQPSafetyFilter
        _HAS_CBF = True
    except ImportError:
        _HAS_CBF = False


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


class AiInferenceRealPytorchNode(Node):
    def __init__(self):
        super().__init__('ai_inference_real_pytorch_node')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(current_dir, '..', 'models', 'model_sim_5.pth')
        default_model_path = resolve_model_path(default_model_path)

        # --- Parameters ---
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('target_beams', 60)
        self.declare_parameter('ai_speed', 1.5)
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('max_steering_angle', 0.35)
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('raw_drive_topic', '/drive_raw')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('speed_scale', 1.0)
        self.declare_parameter('fixed_speed', 0.0)

        self.declare_parameter('enable_cbf', True)
        self.declare_parameter('enable_smoothing', True)
        self.declare_parameter('alpha_steer', 0.1)
        self.declare_parameter('alpha_speed', 0.1)
        self.declare_parameter('d_min', 0.25)
        self.declare_parameter('cbf_gamma', 3.5)
        self.declare_parameter('a_max_brake', 2.61)
        self.declare_parameter('wheelbase', 0.39)

        self.model_path = resolve_model_path(self.get_parameter('model_path').value)
        self.target_beams = self.get_parameter('target_beams').value
        self.ai_speed = self.get_parameter('ai_speed').value
        self.max_range = self.get_parameter('max_range').value
        self.max_steer = self.get_parameter('max_steering_angle').value
        self.drive_topic = self.get_parameter('drive_topic').value
        self.raw_drive_topic = self.get_parameter('raw_drive_topic').value
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

        self.prev_smooth_speed = 0.0
        self.prev_smooth_steer = 0.0

        # --- Performance Counters ---
        self.ai_exec_time_accum = 0.0
        self.cbf_exec_time_accum = 0.0
        self.frame_count = 0
        self.perf_timer = self.create_timer(1.0, self.timer_perf_callback)

        # --- Initializing CBF Shield ---
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
            else:
                self.get_logger().error("CBFQPSafetyFilter module not found! Cannot enable built-in CBF.")

        # --- Load PyTorch Model ---
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
            except Exception as e:
                self.get_logger().warn(f"Failed to load norm stats: {e}")

        if _HAS_TORCH:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.get_logger().info("[PyTorch] Sử dụng GPU (CUDA).")
            else:
                self.device = torch.device('cpu')
                torch.set_num_threads(1)
                self.get_logger().info("[PyTorch] Sử dụng CPU.")

            if os.path.exists(self.model_path) and self.model_path.endswith('.pth'):
                try:
                    raw_model = DAggerMLP(input_dim=self.target_beams, output_dim=2, dropout=0.1).to(self.device)
                    raw_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                    raw_model.eval()
                    self.model = torch.jit.script(raw_model)
                    self.get_logger().info("[PyTorch] Đã tải mô hình JIT thành công.")
                except Exception as e:
                    self.get_logger().error(f"Failed to load PyTorch model: {e}")

        # --- Publishers & Subscriptions ---
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.raw_drive_pub = self.create_publisher(AckermannDriveStamped, self.raw_drive_topic, 10)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)

    def timer_perf_callback(self):
        """Timer chu kỳ 1s để phân tích và in log hiệu suất."""
        if self.frame_count == 0:
            return
            
        avg_ai_latency = self.ai_exec_time_accum / self.frame_count
        avg_cbf_latency = self.cbf_exec_time_accum / self.frame_count
        
        ai_hz = 1.0 / avg_ai_latency if avg_ai_latency > 0 else 0.0
        cbf_hz = 1.0 / avg_cbf_latency if avg_cbf_latency > 0 else 0.0
        
        system_throughput = self.frame_count
        
        self.get_logger().info(
            f"\n[PERF] Throughput thực tế: {system_throughput} Hz (bị giới hạn bởi /scan)\n"
            f"[PERF] AI Inference - Latency: {avg_ai_latency*1000:.2f} ms | Tốc độ max: {ai_hz:.1f} Hz\n"
            f"[PERF] CBF-QP Filter - Latency: {avg_cbf_latency*1000:.2f} ms | Tốc độ max: {cbf_hz:.1f} Hz"
        )
        
        # Reset counters
        self.frame_count = 0
        self.ai_exec_time_accum = 0.0
        self.cbf_exec_time_accum = 0.0

    def scan_callback(self, msg: LaserScan):
        if self.model is None:
            self.publish_drive(self.drive_pub, 0.0, 0.0)
            self.publish_drive(self.raw_drive_pub, 0.0, 0.0)
            return

        # 1. Preprocess
        preprocessed_scan = self.preprocess_scan(msg)

        # 2. Inference AI (Đo thời gian)
        t_ai_start = time.perf_counter()
        ai_speed, ai_steer = self.run_model_inference(preprocessed_scan)
        t_ai_end = time.perf_counter()
        self.ai_exec_time_accum += (t_ai_end - t_ai_start)

        # 3. Smoothing Filter (EMA)
        if self.enable_smoothing:
            smooth_speed = (1.0 - self.alpha_speed) * self.prev_smooth_speed + self.alpha_speed * ai_speed
            smooth_steer = (1.0 - self.alpha_steer) * self.prev_smooth_steer + self.alpha_steer * ai_steer
            self.prev_smooth_speed = smooth_speed
            self.prev_smooth_steer = smooth_steer
        else:
            smooth_speed = ai_speed
            smooth_steer = ai_steer

        # Publish Lệnh Thô
        self.publish_drive(self.raw_drive_pub, smooth_speed, smooth_steer)

        # 4. CBF Safety Filter (Đo thời gian)
        final_speed, final_steer = smooth_speed, smooth_steer
        
        t_cbf_start = time.perf_counter()
        if self.cbf_filter is not None:
            ranges = np.array(msg.ranges, dtype=np.float32)
            angles = np.arange(len(ranges), dtype=np.float32) * msg.angle_increment + msg.angle_min

            u_nom = np.array([smooth_speed, smooth_steer], dtype=np.float32)
            u_safe = self.cbf_filter.filter(u_nom, ranges, angles)

            final_speed, final_steer = float(u_safe[0]), float(u_safe[1])
        t_cbf_end = time.perf_counter()
        self.cbf_exec_time_accum += (t_cbf_end - t_cbf_start)

        # Publish Lệnh An Toàn
        self.publish_drive(self.drive_pub, final_speed, final_steer)
        
        # Cập nhật số frame
        self.frame_count += 1

    def preprocess_scan(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        crop_limit = math.radians(60.0)
        angles = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min
        mask = (angles >= -crop_limit) & (angles <= crop_limit)

        if not np.any(mask):
            return np.ones(self.target_beams, dtype=np.float32) * self.max_range

        valid_ranges = np.clip(np.where(np.isnan(ranges[mask]) | np.isinf(ranges[mask]), self.max_range, ranges[mask]), 0.0, self.max_range)
        target_angles = np.linspace(-crop_limit, crop_limit, self.target_beams)
        return np.interp(target_angles, angles[mask], valid_ranges)

    def run_model_inference(self, preprocessed_scan):
        if not _HAS_TORCH or self.model is None:
            return 0.0, 0.0
        with torch.no_grad():
            norm_scan = preprocessed_scan / 10.0
            tensor_input = torch.tensor(norm_scan, dtype=torch.float32).unsqueeze(0).to(self.device)
            output = self.model(tensor_input).cpu().squeeze(0).numpy()

        if self.target_mean is not None and self.target_std is not None:
            output = output * self.target_std + self.target_mean

        raw_speed = float(output[0])
        speed = float(np.clip(self.fixed_speed if self.fixed_speed > 0.0 else raw_speed * self.speed_scale, 0.0, self.ai_speed))
        steering_angle = float(np.clip(output[1], -self.max_steer, self.max_steer))
        return speed, steering_angle

    def publish_drive(self, publisher, speed, steering_angle):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AiInferenceRealPytorchNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.publish_drive(node.drive_pub, 0.0, 0.0)
        node.publish_drive(node.raw_drive_pub, 0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()