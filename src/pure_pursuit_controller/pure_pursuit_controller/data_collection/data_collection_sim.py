#!/usr/bin/env python3
"""
data_collection_sim.py
──────────────────────
ROS 2 Node dùng để thu thập dữ liệu huấn luyện Imitation Learning trong môi trường mô phỏng (f1tenth_gym_ros).
Subscribe:
  - `/scan` (sensor_msgs/msg/LaserScan)
  - `/drive` (ackermann_msgs/msg/AckermannDriveStamped)

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

class DataCollectionSimNode(Node):
    def __init__(self):
        super().__init__('data_collection_sim_node')

        # --- 1. Parameters ---
        self.declare_parameter('dataset_path', '/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/rrt_2.csv')
        self.declare_parameter('target_beams', 60)
        self.declare_parameter('buffer_size', 50)
        self.declare_parameter('max_range', 10.0)

        self.dataset_path = self.get_parameter('dataset_path').value
        self.target_beams = self.get_parameter('target_beams').value
        self.buffer_size = self.get_parameter('buffer_size').value
        self.max_range = self.get_parameter('max_range').value

        # Tạo thư mục lưu file nếu chưa có
        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)

        # --- 2. State & Sync Variables ---
        self.latest_drive = None
        self.latest_drive_time = 0.0
        
        self.buffer = []
        self.lock = threading.Lock()
        self.total_saved_samples = 0

        # Tạo header cho file CSV mới nếu file chưa tồn tại
        if not os.path.exists(self.dataset_path):
            self._write_header()

        # --- 3. Pub/Sub ---
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.drive_sub = self.create_subscription(AckermannDriveStamped, '/drive', self.drive_callback, 10)

        self.get_logger().info("=========================================")
        self.get_logger().info(" DATA COLLECTION SIM NODE STARTED")
        self.get_logger().info(f" Dataset Path: {self.dataset_path}")
        self.get_logger().info(f" Target Beams: {self.target_beams} | Buffer Size: {self.buffer_size}")
        self.get_logger().info("=========================================")

    def _write_header(self):
        """Khởi tạo header cho file CSV"""
        with open(self.dataset_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [f'lidar_{i}' for i in range(self.target_beams)] + ['speed', 'steering_angle']
            writer.writerow(header)
        self.get_logger().info("Created new simulation dataset CSV file with headers.")

    def drive_callback(self, msg: AckermannDriveStamped):
        """Lưu lại lệnh điều khiển mới nhất để đồng bộ với scan"""
        with self.lock:
            self.latest_drive = msg
            self.latest_drive_time = time.monotonic()

    def scan_callback(self, msg: LaserScan):
        """Xử lý scan, đồng bộ hóa và lưu vào buffer"""
        now = time.monotonic()
        
        # Kiểm tra sự tồn tại và tính hợp lệ thời gian của lệnh lái (không quá 0.5 giây)
        with self.lock:
            if self.latest_drive is None or (now - self.latest_drive_time) > 0.5:
                return
            current_drive = self.latest_drive
            speed = current_drive.drive.speed
            steering_angle = current_drive.drive.steering_angle

        # Tiền xử lý scan (Crop & Downsample)
        preprocessed_scan = self.preprocess_scan(msg)

        # Lưu dữ liệu vào buffer
        with self.lock:
            row = list(preprocessed_scan) + [speed, steering_angle]
            self.buffer.append(row)
            
            # Đạt ngưỡng buffer_size thì flush xuống disk
            if len(self.buffer) >= self.buffer_size:
                buffer_to_save = list(self.buffer)
                self.buffer.clear()
                
                # Chạy luồng ghi file phụ để tránh block callback chính
                threading.Thread(target=self._flush_buffer, args=(buffer_to_save,), daemon=True).start()

    def preprocess_scan(self, msg: LaserScan):
        """
        Crop góc quét LiDAR về [-60, 60] độ phía trước mặt
        và downsample về đúng self.target_beams bằng cách nội suy.
        """
        ranges = np.array(msg.ranges) # 1080
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment

        # Giới hạn góc quét (radians)
        crop_limit = math.radians(60.0)

        # Tính toán góc tương ứng với từng điểm scan
        angles = np.arange(len(ranges)) * angle_increment + angle_min

        # Lọc ra các điểm nằm trong góc [-60, 60] độ
        mask = (angles >= -crop_limit) & (angles <= crop_limit)
        
        if not np.any(mask):
            return np.ones(self.target_beams, dtype=np.float32) * self.max_range

        valid_ranges = ranges[mask]
        valid_angles = angles[mask]

        # Xử lý các giá trị NaN và Vô cùng (inf)
        valid_ranges = np.where(np.isnan(valid_ranges) | np.isinf(valid_ranges), self.max_range, valid_ranges)
        valid_ranges = np.clip(valid_ranges, 0.0, self.max_range)

        # Nội suy (Interpolation) để có đúng target_beams
        target_angles = np.linspace(-crop_limit, crop_limit, self.target_beams)
        preprocessed_ranges = np.interp(target_angles, valid_angles, valid_ranges)

        return preprocessed_ranges

    def _flush_buffer(self, data_list):
        """Ghi dữ liệu từ memory xuống file CSV (Thread-safe)"""
        try:
            with open(self.dataset_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data_list)
            
            self.total_saved_samples += len(data_list)
            self.get_logger().info(f"[SIM] Flushed {len(data_list)} samples to CSV. Total samples: {self.total_saved_samples}")
        except Exception as e:
            self.get_logger().error(f"Error while flushing buffer to CSV: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectionSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Shutting down simulation data collection node.")
        # Flush nốt dữ liệu còn sót trong buffer trước khi dừng
        if len(node.buffer) > 0:
            node._flush_buffer(node.buffer)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
