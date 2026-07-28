#!/usr/bin/env python3
"""
ai_inference_sim.py
───────────────────
ROS 2 Node chạy Suy luận mô hình AI tự lái thuần túy (Autonomous Only) bằng ONNX Runtime.
Không thực hiện ghi đè từ chuyên gia (No override) và không thu thập dữ liệu DAgger.

Logic:
  1. Đọc và giải mã dữ liệu LiDAR từ `/scan`, tiền xử lý về 60 beams.
  2. Suy luận qua mô hình ONNX để dự đoán [tốc độ, góc lái].
     (Model ONNX này được export bằng export_onnx.py / DeployWrapper, nên đã tự
     chuẩn hóa input và denormalize output bên trong đồ thị -> KHÔNG cần tự
     chia/nhân lại ở đây. input_name = 'lidar_raw' là dấu hiệu nhận biết.)
  3. Publish lệnh điều khiển trực tiếp tới `/drive`.
"""

import os
os.environ["ORT_DISABLE_CPU_AFFINITY"] = "1"
import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

try:
    import onnxruntime as ort
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False


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


class AiInferenceSimNode(Node):
    def __init__(self):
        super().__init__('ai_inference_sim_node')

        # --- 1. Parameters ---
        default_model = '/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/models/rrt_2.onnx'
        if not os.path.exists(default_model):
            default_model = resolve_model_path(default_model)

        self.declare_parameter('model_path', default_model)
        self.declare_parameter('target_beams', 60)
        self.declare_parameter('ai_speed', 10.0)           # Vận tốc tối đa của AI (m/s)
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('scan_topic', 'scan_raw')
        self.declare_parameter('speed_scale', 1.0)        # Tỷ lệ nhân tốc độ (vd: 1.5 = tăng 50%)
        self.declare_parameter('fixed_speed', 0.0)        # Nếu > 0, ép tốc độ chạy cố định (vd: 5.5 m/s)

        self.model_path = resolve_model_path(self.get_parameter('model_path').value)
        self.target_beams = self.get_parameter('target_beams').value
        self.ai_speed = self.get_parameter('ai_speed').value
        self.max_range = self.get_parameter('max_range').value
        self.drive_topic = self.get_parameter('drive_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.speed_scale = self.get_parameter('speed_scale').value
        self.fixed_speed = self.get_parameter('fixed_speed').value

        # --- 2. Load ONNX Model ---
        self.ort_session = None
        self.input_name = 'input'
        self.target_mean = None
        self.target_std = None

        # Tự động tìm file lưu tham số chuẩn hóa nếu có
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

        if _HAS_ONNX:
            if os.path.exists(self.model_path) and self.model_path.endswith('.onnx'):
                try:
                    self.ort_session = ort.InferenceSession(self.model_path)
                    self.input_name = self.ort_session.get_inputs()[0].name
                    self.get_logger().info(f"Successfully loaded ONNX model from {self.model_path} (input: '{self.input_name}')")
                    if self.input_name != 'lidar_raw':
                        self.get_logger().warn(
                            f"Input name '{self.input_name}' != 'lidar_raw'. Model này có thể KHÔNG "
                            f"phải export bằng DeployWrapper (export_onnx.py) -> Sẽ tự chuẩn hóa "
                            f"đầu vào và giải chuẩn hóa (denormalize) đầu ra trong code Python."
                        )
                except Exception as e:
                    self.get_logger().error(f"Failed to load ONNX model: {e}")
            else:
                self.get_logger().warn(f"ONNX Model file not found at {self.model_path} or invalid format.")
        else:
            self.get_logger().error("ONNX Runtime is not installed. Install it with: pip install onnxruntime")

        # --- 3. Pub/Sub ---
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)

        self.get_logger().info("=========================================")
        self.get_logger().info(" AI AUTONOMOUS INFERENCE SIM NODE STARTED")
        self.get_logger().info(f" Model Path: {self.model_path}")
        self.get_logger().info(f" Target Speed: {self.ai_speed} m/s")
        self.get_logger().info("=========================================")

    def scan_callback(self, msg: LaserScan):
        """Xử lý LiDAR, suy luận qua mạng AI và điều khiển xe"""
        if self.ort_session is None:
            # Nếu chưa có model, dừng xe
            self.publish_drive(0.0, 0.0)
            self.get_logger().warn("ONNX model is not loaded. Car stopped.", throttle_duration_sec=2.0)
            return

        # 1. Tiền xử lý dữ liệu scan (Crop góc quét & downsample)
        preprocessed_scan = self.preprocess_scan(msg)

        # 2. Suy luận qua mô hình AI
        ai_speed, ai_steer = self.run_model_inference(preprocessed_scan)

        # 3. Điều khiển xe chạy tự động
        self.publish_drive(ai_speed, ai_steer)
        self.get_logger().info(f"[AI DRIVING] Speed: {ai_speed:.2f} m/s | Steer: {math.degrees(ai_steer):.1f}°", throttle_duration_sec=1.0)

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
        """Dự đoán lệnh lái Ackermann qua ONNX Model"""
        if not _HAS_ONNX or self.ort_session is None:
            return 0.0, 0.0

        # Nếu model không phải DeployWrapper, ta cần chia 10.0 cho input LiDAR ở đây
        if self.input_name != 'lidar_raw':
            norm_scan = (preprocessed_scan / 10.0).astype(np.float32)
        else:
            norm_scan = preprocessed_scan.astype(np.float32)

        tensor_input = np.expand_dims(norm_scan, axis=0)

        # Chạy suy luận qua ONNX Runtime
        outputs = self.ort_session.run(None, {self.input_name: tensor_input})
        output = outputs[0][0]  # [speed, steering_angle]

        # Giải chuẩn hóa (Denormalize) target nếu có file _norm.json đi kèm
        if self.input_name != 'lidar_raw' and self.target_mean is not None and self.target_std is not None:
            output = output * self.target_std + self.target_mean

        # Output: [speed, steering_angle]
        raw_speed = float(output[0])
        if self.fixed_speed > 0.0:
            speed = float(np.clip(self.fixed_speed, 0.0, self.ai_speed))
        else:
            speed = float(np.clip(raw_speed * self.speed_scale, 0.0, self.ai_speed))
        steering_angle = float(np.clip(output[1], -0.41, 0.41))
        return speed, steering_angle

    def publish_drive(self, speed, steering_angle):
        """Publish lệnh tới topic /drive (AckermannDriveStamped)"""
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'ego_racecar'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AiInferenceSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Shutting down AI simulation inference node.")
        node.publish_drive(0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
