#!/usr/bin/env python3
"""
data_collection_sim_20hz.py
───────────────────────────
ROS 2 Node tối giản dùng để thu thập dữ liệu trong Mô phỏng ở tần số đúng 20 Hz.
Không sử dụng bàn phím raw mode -> Ctrl+C nhạy 100%.

Format CSV: [lidar_0, lidar_1, ..., lidar_59, speed, steering_angle]
"""

import os
import csv
import math
import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class DataCollectionSim20HzNode(Node):
    def __init__(self):
        super().__init__('data_collection_sim_20hz_node')

        # --- 1. Parameters ---
        default_path = '/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/datasets/rrt_20hz_sim.csv'
        self.declare_parameter('dataset_path', default_path)
        self.declare_parameter('target_beams', 60)
        self.declare_parameter('buffer_size', 50)
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('sample_rate', 20.0)  # Hãm tần số 20 Hz (0.05s một mẫu)

        self.dataset_path = self.get_parameter('dataset_path').value
        self.target_beams = self.get_parameter('target_beams').value
        self.buffer_size = self.get_parameter('buffer_size').value
        self.max_range = self.get_parameter('max_range').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.sample_interval = 1.0 / self.sample_rate  # 0.05s

        # Tạo thư mục lưu file nếu chưa có
        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)

        # --- 2. State & Sync Variables ---
        self.latest_drive = None
        self.latest_drive_time = 0.0
        self.last_save_time = 0.0

        self.buffer = []
        self.lock = threading.Lock()
        self.total_saved_samples = 0

        # Tạo header cho file CSV mới nếu file chưa tồn tại
        if not os.path.exists(self.dataset_path):
            self._write_header()

        # --- 3. Subscriptions ---
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.drive_sub = self.create_subscription(AckermannDriveStamped, '/drive', self.drive_callback, 10)

        self.get_logger().info("=========================================")
        self.get_logger().info(" ⏱️  CLEAN 20 Hz SIM DATA COLLECTOR STARTED")
        self.get_logger().info(f" Dataset Path : {self.dataset_path}")
        self.get_logger().info(f" Rate         : {self.sample_rate} Hz (Interval: {self.sample_interval:.3f}s)")
        self.get_logger().info(" Press Ctrl + C anytime to stop and save.")
        self.get_logger().info("=========================================")

    def _write_header(self):
        with open(self.dataset_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [f'lidar_{i}' for i in range(self.target_beams)] + ['speed', 'steering_angle']
            writer.writerow(header)
        self.get_logger().info("Created new CSV file with headers.")

    def drive_callback(self, msg: AckermannDriveStamped):
        with self.lock:
            self.latest_drive = msg
            self.latest_drive_time = time.monotonic()

    def scan_callback(self, msg: LaserScan):
        now = time.monotonic()

        # 🚀 HÃM TẦN SỐ VỀ ĐÚNG 20 Hz (0.05s một mẫu)
        if (now - self.last_save_time) < self.sample_interval:
            return

        # Kiểm tra tính hợp lệ thời gian của lệnh lái (không quá 0.5s cũ)
        with self.lock:
            if self.latest_drive is None or (now - self.latest_drive_time) > 0.5:
                return
            current_drive = self.latest_drive
            speed = current_drive.drive.speed
            steering_angle = current_drive.drive.steering_angle

        # Đánh dấu thời điểm lưu mẫu
        self.last_save_time = now

        # Tiền xử lý scan (Crop & Downsample về 60 beams)
        preprocessed_scan = self.preprocess_scan(msg)

        # Lưu dữ liệu vào buffer
        with self.lock:
            row = list(preprocessed_scan) + [speed, steering_angle]
            self.buffer.append(row)
            
            if len(self.buffer) >= self.buffer_size:
                buffer_to_save = list(self.buffer)
                self.buffer.clear()
                threading.Thread(target=self._flush_buffer, args=(buffer_to_save,), daemon=True).start()

    def preprocess_scan(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
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
        resampled_ranges = np.interp(target_angles, valid_angles, valid_ranges)

        return resampled_ranges.astype(np.float32)

    def _flush_buffer(self, data_list):
        try:
            with open(self.dataset_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data_list)
            with self.lock:
                self.total_saved_samples += len(data_list)
                total = self.total_saved_samples
            self.get_logger().info(f" 💾 Saved +{len(data_list)} samples (20Hz). Total in CSV: {total}")
        except Exception as e:
            self.get_logger().error(f"Failed to flush buffer to disk: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectionSim20HzNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Received SIGINT (Ctrl+C). Shutting down...")
    finally:
        with node.lock:
            if len(node.buffer) > 0:
                node._flush_buffer(node.buffer)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
