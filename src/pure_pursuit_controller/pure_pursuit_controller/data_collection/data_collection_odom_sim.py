#!/usr/bin/env python3
"""
data_collection_odom_sim.py
───────────────────────────
ROS 2 Node dùng để thu thập dữ liệu huấn luyện Imitation Learning dựa trên ODOMETRY trong môi trường mô phỏng (f1tenth_gym_ros).
Subscribe:
  - `/ego_racecar/odom` (nav_msgs/msg/Odometry) - Vị trí, hướng góc, vận tốc của xe
  - `/drive` (ackermann_msgs/msg/AckermannDriveStamped) - Lệnh lái chuyên gia (tốc độ và góc lái)

Format CSV: [x, y, z, yaw, qx, qy, qz, qw, linear_vx, linear_vy, angular_wz, speed, steering_angle]
"""

import os
import csv
import math
import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class DataCollectionOdomSimNode(Node):
    def __init__(self):
        super().__init__('data_collection_odom_sim_node')

        # --- 1. Parameters ---
        # Xác định đường dẫn mặc định linh hoạt cho cả Docker (/sim_ws) lẫn máy Host (~/f1_ws)
        if os.path.exists('/sim_ws'):
            default_dataset_path = '/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/odom_dataset_sim.csv'
        else:
            home_dir = os.path.expanduser('~')
            default_dataset_path = os.path.join(
                home_dir,
                'f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/odom_dataset_sim.csv'
            )

        self.declare_parameter('dataset_path', default_dataset_path)
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('buffer_size', 50)

        self.dataset_path = self.get_parameter('dataset_path').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.drive_topic = self.get_parameter('drive_topic').value
        self.buffer_size = self.get_parameter('buffer_size').value

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
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.drive_sub = self.create_subscription(AckermannDriveStamped, self.drive_topic, self.drive_callback, 10)

        self.get_logger().info("=========================================")
        self.get_logger().info(" DATA COLLECTION ODOM SIM NODE STARTED")
        self.get_logger().info(f" Dataset Path: {self.dataset_path}")
        self.get_logger().info(f" Odom Topic: {self.odom_topic} | Drive Topic: {self.drive_topic}")
        self.get_logger().info(f" Buffer Size: {self.buffer_size}")
        self.get_logger().info("=========================================")

    def _write_header(self):
        """Khởi tạo header cho file CSV dữ liệu Odometry"""
        with open(self.dataset_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [
                'x', 'y', 'z', 
                'yaw', 
                'qx', 'qy', 'qz', 'qw', 
                'linear_vx', 'linear_vy', 'angular_wz', 
                'speed', 'steering_angle'
            ]
            writer.writerow(header)
        self.get_logger().info("Created new simulation ODOMETRY dataset CSV file with headers.")

    def drive_callback(self, msg: AckermannDriveStamped):
        """Lưu lại lệnh điều khiển mới nhất để đồng bộ với dữ liệu Odometry"""
        with self.lock:
            self.latest_drive = msg
            self.latest_drive_time = time.monotonic()

    def odom_callback(self, msg: Odometry):
        """Xử lý Odometry, đồng bộ hóa với lệnh drive và lưu vào buffer"""
        now = time.monotonic()

        # Kiểm tra sự tồn tại và tính hợp lệ thời gian của lệnh lái (không quá 0.5 giây)
        with self.lock:
            if self.latest_drive is None or (now - self.latest_drive_time) > 0.5:
                return
            current_drive = self.latest_drive
            speed = current_drive.drive.speed
            steering_angle = current_drive.drive.steering_angle

        # Trích xuất dữ liệu vị trí (Position)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        # Trích xuất dữ liệu hướng góc (Orientation Quaternion)
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        # Tính góc Yaw (radians) từ Quaternion
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # Trích xuất vận tốc (Twist)
        linear_vx = msg.twist.twist.linear.x
        linear_vy = msg.twist.twist.linear.y
        angular_wz = msg.twist.twist.angular.z

        # Lưu dữ liệu vào buffer
        with self.lock:
            row = [x, y, z, yaw, qx, qy, qz, qw, linear_vx, linear_vy, angular_wz, speed, steering_angle]
            self.buffer.append(row)

            # Đạt ngưỡng buffer_size thì flush xuống disk
            if len(self.buffer) >= self.buffer_size:
                buffer_to_save = list(self.buffer)
                self.buffer.clear()

                # Chạy luồng ghi file phụ để tránh block callback chính
                threading.Thread(target=self._flush_buffer, args=(buffer_to_save,), daemon=True).start()

    def _flush_buffer(self, data_list):
        """Ghi dữ liệu từ memory xuống file CSV (Thread-safe)"""
        try:
            with open(self.dataset_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data_list)

            self.total_saved_samples += len(data_list)
            self.get_logger().info(f"[ODOM SIM] Flushed {len(data_list)} samples to CSV. Total samples: {self.total_saved_samples}")
        except Exception as e:
            self.get_logger().error(f"Error while flushing buffer to CSV: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectionOdomSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Shutting down simulation Odom data collection node.")
        # Flush nốt dữ liệu còn sót trong buffer trước khi dừng
        if len(node.buffer) > 0:
            node._flush_buffer(node.buffer)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
