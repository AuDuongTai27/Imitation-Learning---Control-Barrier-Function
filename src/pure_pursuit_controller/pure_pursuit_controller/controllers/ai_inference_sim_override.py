#!/usr/bin/env python3
"""
ai_inference_sim_override.py
────────────────────────────
ROS 2 Node running AI inference with expert Pure Pursuit override mechanism in simulation.
"""

import os
os.environ["ORT_DISABLE_CPU_AFFINITY"] = "1"
import csv
import math
import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

try:
    import onnxruntime as ort
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False


class AiInferenceSimOverrideNode(Node):
    def __init__(self):
        super().__init__('ai_inference_sim_override_node')

        # --- 1. Parameters ---
        self.declare_parameter('model_path', '/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/dagger_model_sim_4.onnx')
        self.declare_parameter('waypoint_path', '/sim_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv')
        self.declare_parameter('dataset_path', '/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/dagger_dataset_sim_4.csv')
        
        self.declare_parameter('target_beams', 60)
        self.declare_parameter('lookahead_dist', 1.0)
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('ai_speed', 3.0)
        self.declare_parameter('cte_threshold', 0.5)
        self.declare_parameter('buffer_size', 50)
        self.declare_parameter('max_range', 10.0)

        self.model_path = self.get_parameter('model_path').value
        self.waypoint_path = self.get_parameter('waypoint_path').value
        self.dataset_path = self.get_parameter('dataset_path').value
        
        self.target_beams = self.get_parameter('target_beams').value
        self.L = self.get_parameter('lookahead_dist').value
        self.max_speed = self.get_parameter('max_speed').value
        self.ai_speed = self.get_parameter('ai_speed').value
        self.cte_threshold = self.get_parameter('cte_threshold').value
        self.buffer_size = self.get_parameter('buffer_size').value
        self.max_range = self.get_parameter('max_range').value

        # --- 2. Load ONNX Model ---
        self.ort_session = None
        self.input_name = 'input'
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

        if _HAS_ONNX:
            if os.path.exists(self.model_path) and self.model_path.endswith('.onnx'):
                try:
                    self.ort_session = ort.InferenceSession(self.model_path)
                    self.input_name = self.ort_session.get_inputs()[0].name
                    self.get_logger().info(f"Successfully loaded ONNX model from {self.model_path} (input: '{self.input_name}')")
                except Exception as e:
                    self.get_logger().error(f"Failed to load ONNX model: {e}")
            else:
                self.get_logger().warn(f"ONNX Model file not found at {self.model_path}. Running in Expert mode.")
        else:
            self.get_logger().error("ONNX Runtime is not installed.")

        # --- 3. Waypoint & Pure Pursuit States ---
        self.waypoints = self.load_waypoints(self.waypoint_path)
        self.last_idx = 0
        
        # --- 4. State & Buffer Variables ---
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
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)

        self.get_logger().info("=========================================")
        self.get_logger().info(" AI INFERENCE WITH OVERRIDE NODE STARTED")
        self.get_logger().info(f" Model Path: {self.model_path}")
        self.get_logger().info(f" CTE Override Threshold: {self.cte_threshold} m")
        self.get_logger().info(f" Dataset Path: {self.dataset_path}")
        self.get_logger().info("=========================================")

    def load_waypoints(self, file_path):
        """Load waypoints from CSV file"""
        if os.path.exists(file_path):
            try:
                points = []
                with open(file_path, 'r') as f:
                    reader = csv.reader(f)
                    first_row = next(reader, None)
                    if first_row:
                        try:
                            points.append([float(first_row[0]), float(first_row[1])])
                        except ValueError:
                            pass
                    for row in reader:
                        if len(row) >= 2:
                            points.append([float(row[0]), float(row[1])])
                self.get_logger().info(f"Loaded {len(points)} waypoints from CSV.")
                return np.array(points)
            except Exception as e:
                self.get_logger().error(f"Error loading waypoints: {e}")

        # Fallback: circular path
        self.get_logger().warn("Waypoint file not found! Generating circular fallback waypoints.")
        theta = np.linspace(0, 2*np.pi, 200)
        r = 4.0
        points = np.stack([r * np.cos(theta), r * np.sin(theta) + 3.0], axis=1)
        return points

    def _write_header(self):
        with open(self.dataset_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [f'lidar_{i}' for i in range(self.target_beams)] + ['speed', 'steering_angle']
            writer.writerow(header)
        self.get_logger().info("Created simulation dataset CSV file.")

    def odom_callback(self, msg: Odometry):
        with self.lock:
            self.car_x = msg.pose.pose.position.x
            self.car_y = msg.pose.pose.position.y
            
            q = msg.pose.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.car_yaw = math.atan2(siny_cosp, cosy_cosp)
            self.odom_received = True

    def scan_callback(self, msg: LaserScan):
        if not self.odom_received:
            return

        with self.lock:
            curr_x = self.car_x
            curr_y = self.car_y
            curr_yaw = self.car_yaw

        num_pts = len(self.waypoints)
        search_len = 50 
        indices = [(self.last_idx + i) % num_pts for i in range(search_len)]
        search_points = np.array([self.waypoints[i] for i in indices])
        dists = np.linalg.norm(search_points - np.array([curr_x, curr_y]), axis=1)
        min_local_idx = np.argmin(dists)
        nearest_idx = indices[min_local_idx]
        self.last_idx = nearest_idx

        cte = dists[min_local_idx]

        lookahead_idx = nearest_idx
        target_point = self.waypoints[nearest_idx]
        while True:
            lookahead_idx = (lookahead_idx + 1) % num_pts
            dist = math.dist([curr_x, curr_y], self.waypoints[lookahead_idx])
            if dist >= self.L or lookahead_idx == nearest_idx:
                target_point = self.waypoints[lookahead_idx]
                break

        expert_steer = self.calculate_steering(target_point, curr_x, curr_y, curr_yaw)
        expert_steer = np.clip(expert_steer, -0.35, 0.35)
        expert_speed = self.max_speed

        preprocessed_scan = self.preprocess_scan(msg)

        if self.ort_session is None or cte >= self.cte_threshold:
            self.publish_drive(expert_speed, expert_steer)
            self.get_logger().warn(
                f"[OVERRIDE] CTE: {cte:.3f}m >= {self.cte_threshold}m | Expert Steer: {math.degrees(expert_steer):.1f}°", 
                throttle_duration_sec=1.0
            )
            
            with self.lock:
                row = list(preprocessed_scan) + [expert_speed, expert_steer]
                self.buffer.append(row)
                
                if len(self.buffer) >= self.buffer_size:
                    buffer_to_save = list(self.buffer)
                    self.buffer.clear()
                    threading.Thread(target=self._flush_buffer, args=(buffer_to_save,), daemon=True).start()
        else:
            ai_speed, ai_steer = self.run_model_inference(preprocessed_scan)
            self.publish_drive(ai_speed, ai_steer)
            self.get_logger().info(
                f"[AI DRIVING] CTE: {cte:.3f}m < {self.cte_threshold}m | AI Steer: {math.degrees(ai_steer):.1f}°", 
                throttle_duration_sec=1.5
            )

    def calculate_steering(self, target, car_x, car_y, car_yaw):
        tx, ty = target
        dx = tx - car_x
        dy = ty - car_y
        
        target_y_local = dx * math.sin(-car_yaw) + dy * math.cos(-car_yaw)
        lookahead_dist = math.dist([car_x, car_y], target)
        wheelbase = 0.33
        
        return math.atan((2 * wheelbase * target_y_local) / (lookahead_dist**2))

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

        if self.input_name != 'lidar_raw':
            norm_scan = (preprocessed_scan / 10.0).astype(np.float32)
        else:
            norm_scan = preprocessed_scan.astype(np.float32)

        tensor_input = np.expand_dims(norm_scan, axis=0)

        outputs = self.ort_session.run(None, {self.input_name: tensor_input})
        output = outputs[0][0]

        if self.input_name != 'lidar_raw' and self.target_mean is not None and self.target_std is not None:
            output = output * self.target_std + self.target_mean

        speed = float(np.clip(output[0], 0.0, self.ai_speed))
        steering_angle = float(np.clip(output[1], -0.35, 0.35))
        return speed, steering_angle

    def publish_drive(self, speed, steering_angle):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'ego_racecar'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)

    def _flush_buffer(self, data_list):
        try:
            with open(self.dataset_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data_list)
            
            self.total_saved_samples += len(data_list)
            self.get_logger().info(f"[OVERRIDE LOGGING] Saved {len(data_list)} samples. Total: {self.total_saved_samples}")
        except Exception as e:
            self.get_logger().error(f"Error saving dataset: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = AiInferenceSimOverrideNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Shutting down AI Simulation inference with override node.")
        node.publish_drive(0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
