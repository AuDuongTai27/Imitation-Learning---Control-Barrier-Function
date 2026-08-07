#!/usr/bin/env python3
"""
ai_inference_real.py
───────────────────
ROS 2 Node chạy Suy luận mô hình AI tự lái thuần túy (Autonomous Only) trên XE THẬT Jetson.
Nạp file mô hình ONNX nằm cùng cấp thư mục với script này.
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


class AiInferenceRealNode(Node):
    def __init__(self):
        super().__init__('ai_inference_real_node')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(current_dir, '..', 'models', 'dagger_model_sim.onnx')
        default_model_path = resolve_model_path(default_model_path)

        # --- 1. Parameters ---
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('target_beams', 60)
        self.declare_parameter('ai_speed', 1.0)           # Speed limit (m/s)
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('max_steering_angle', 0.35) # Max steering angle (rad)

        self.model_path = resolve_model_path(self.get_parameter('model_path').value)
        self.target_beams = self.get_parameter('target_beams').value
        self.ai_speed = self.get_parameter('ai_speed').value
        self.max_range = self.get_parameter('max_range').value
        self.max_steer = self.get_parameter('max_steering_angle').value

        # --- 2. Load ONNX Model ---
        self.ort_session = None
        if _HAS_ONNX:
            if os.path.exists(self.model_path) and self.model_path.endswith('.onnx'):
                try:
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = 1
                    options.inter_op_num_threads = 1
                    try:
                        options.add_session_config_entry("session.set_affinity", "0")
                    except AttributeError:
                        pass
                    
                    self.ort_session = ort.InferenceSession(self.model_path, sess_options=options)
                    self.get_logger().info(f"Successfully loaded ONNX model from {self.model_path}")
                except Exception as e:
                    self.get_logger().error(f"Failed to load ONNX model: {e}")
            else:
                self.get_logger().error(f"ONNX Model file not found at {self.model_path}!")
        else:
            self.get_logger().error("ONNX Runtime is not installed.")

        # --- 3. Pub/Sub ---
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.get_logger().info("=========================================")
        self.get_logger().info(" AI AUTONOMOUS INFERENCE (REAL VEHICLE) READY")
        self.get_logger().info(f" Model Path: {self.model_path}")
        self.get_logger().info(f" Safe Speed Limit: {self.ai_speed} m/s")
        self.get_logger().info(f" Max Steer Limit: {self.max_steer} rad")
        self.get_logger().info("=========================================")

    def scan_callback(self, msg: LaserScan):
        if self.ort_session is None:
            self.publish_drive(0.0, 0.0)
            self.get_logger().warn("ONNX model is not loaded. Car stopped.", throttle_duration_sec=2.0)
            return

        preprocessed_scan = self.preprocess_scan(msg)
        ai_speed, ai_steer = self.run_model_inference(preprocessed_scan)

        self.publish_drive(ai_speed, ai_steer)
        self.get_logger().info(f"[REAL - AI] Speed: {ai_speed:.2f} m/s | Steer: {math.degrees(ai_steer):.1f}°", throttle_duration_sec=1.0)

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
        """Inference steering and speed from ONNX model"""
        if not _HAS_ONNX or self.ort_session is None:
            return 0.0, 0.0

        norm_scan = (preprocessed_scan / 10.0).astype(np.float32)
        tensor_input = np.expand_dims(norm_scan, axis=0)

        outputs = self.ort_session.run(None, {'input': tensor_input})
        output = outputs[0][0]

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
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AiInferenceRealNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Shutting down AI real vehicle inference node.")
        node.publish_drive(0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
