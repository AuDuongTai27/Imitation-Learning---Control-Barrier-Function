#!/usr/bin/env python3
"""
ai_inference_sim_odom.py
────────────────────────
ROS 2 Node for odometry-based AI inference in simulation using ONNX Runtime.
"""

import os
os.environ["ORT_DISABLE_CPU_AFFINITY"] = "1"
import math
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

try:
    import onnxruntime as ort
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False


INPUT_COLS = ['x', 'y', 'yaw', 'linear_vx', 'linear_vy', 'angular_wz']


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


class AiInferenceSimOdomNode(Node):
    def __init__(self):
        super().__init__('ai_inference_sim_odom_node')

        # --- 1. Parameters ---
        if os.path.exists('/sim_ws'):
            default_model_path = '/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/models/odom_replay_model.onnx'
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            default_model_path = os.path.join(current_dir, '..', 'models', 'odom_replay_model.onnx')
        default_model_path = resolve_model_path(default_model_path)

        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('target_beams', 60)
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('ai_speed', 3.0)           # Max AI speed (m/s)
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('max_steering_angle', 0.41)

        self.model_path = resolve_model_path(self.get_parameter('model_path').value)
        self.odom_topic = self.get_parameter('odom_topic').value
        self.drive_topic = self.get_parameter('drive_topic').value
        self.ai_speed = self.get_parameter('ai_speed').value
        self.max_steer = self.get_parameter('max_steering_angle').value

        # --- 2. Load ONNX Model ---
        self.ort_session = None
        self.input_name = 'input'

        self.target_mean = None
        self.target_std = None
        self.input_mean = None
        self.input_std = None

        norm_path = os.path.splitext(self.model_path)[0] + '_norm.json'
        if os.path.exists(norm_path):
            try:
                import json
                with open(norm_path, 'r') as f:
                    stats = json.load(f)
                if 'target_mean' in stats and 'target_std' in stats:
                    self.target_mean = np.array(stats['target_mean'], dtype=np.float32)
                    self.target_std = np.array(stats['target_std'], dtype=np.float32)
                if 'input_mean' in stats and 'input_std' in stats:
                    self.input_mean = np.array(stats['input_mean'], dtype=np.float32)
                    self.input_std = np.array(stats['input_std'], dtype=np.float32)
                self.get_logger().info(f"Loaded normalization stats from {norm_path}")
            except Exception as e:
                self.get_logger().warn(f"Failed to load normalization stats: {e}")

        if _HAS_ONNX:
            if os.path.exists(self.model_path) and self.model_path.endswith('.onnx'):
                try:
                    self.ort_session = ort.InferenceSession(self.model_path)
                    self.input_name = self.ort_session.get_inputs()[0].name
                    self.get_logger().info(
                        f"Successfully loaded ODOM ONNX model from {self.model_path} (input: '{self.input_name}')"
                    )
                except Exception as e:
                    self.get_logger().error(f"Failed to load ONNX model: {e}")
            else:
                self.get_logger().warn(f"ONNX Model file not found at {self.model_path} or invalid format.")
        else:
            self.get_logger().error("ONNX Runtime is not installed.")

        # --- 3. Pub/Sub ---
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)

        self.get_logger().info("=========================================")
        self.get_logger().info(" AI ODOMETRY INFERENCE SIM NODE STARTED")
        self.get_logger().info(f" Model Path: {self.model_path}")
        self.get_logger().info(f" Odom Topic: {self.odom_topic} | Drive Topic: {self.drive_topic}")
        self.get_logger().info(f" Input order: {INPUT_COLS}")
        self.get_logger().info(f" Target Speed Limit: {self.ai_speed} m/s")
        self.get_logger().info("=========================================")

    def odom_callback(self, msg: Odometry):
        if self.ort_session is None:
            self.publish_drive(0.0, 0.0)
            self.get_logger().warn("ONNX model is not loaded. Car stopped.", throttle_duration_sec=2.0)
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        linear_vx = msg.twist.twist.linear.x
        linear_vy = msg.twist.twist.linear.y
        angular_wz = msg.twist.twist.angular.z

        features = np.array([x, y, yaw, linear_vx, linear_vy, angular_wz], dtype=np.float32)
        ai_speed, ai_steer = self.run_model_inference(features)

        self.publish_drive(ai_speed, ai_steer)
        self.get_logger().info(
            f"[AI ODOM DRIVING] Speed: {ai_speed:.2f} m/s | Steer: {math.degrees(ai_steer):.1f}°",
            throttle_duration_sec=1.0
        )

    def run_model_inference(self, features: np.ndarray):
        """Inference steering and speed from ONNX Model"""
        if not _HAS_ONNX or self.ort_session is None:
            return 0.0, 0.0

        if self.input_name != 'odom_state' and self.input_mean is not None and self.input_std is not None:
            features = (features - self.input_mean) / (self.input_std + 1e-8)

        tensor_input = np.expand_dims(features, axis=0).astype(np.float32)

        outputs = self.ort_session.run(None, {self.input_name: tensor_input})
        output = outputs[0][0]

        if self.input_name != 'odom_state' and self.target_mean is not None and self.target_std is not None:
            output = output * self.target_std + self.target_mean

        speed = float(np.clip(output[0], 0.0, self.ai_speed))
        steering_angle = float(np.clip(output[1], -self.max_steer, self.max_steer))
        return speed, steering_angle

    def publish_drive(self, speed, steering_angle):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'ego_racecar'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AiInferenceSimOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Shutting down AI Odom simulation inference node.")
        node.publish_drive(0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()