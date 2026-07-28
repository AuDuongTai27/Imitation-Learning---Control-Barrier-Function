#!/usr/bin/env python3
"""
steering_logger.py
──────────────────
ROS 2 Node ghi lại dữ liệu góc lái ra lệnh (/drive) và phản hồi thực tế (/odom)
của xe thật để kiểm tra độ trễ (latency) và phản ứng của hệ thống cơ cấu lái.

Dữ liệu được lưu vào file CSV: `steering_log.csv` nằm cùng thư mục.
"""

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import csv
import os
import time

class SteeringLoggerNode(Node):
    def __init__(self):
        super().__init__('steering_logger_node')
        
        # --- Đường dẫn lưu file CSV ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(current_dir, 'steering_log.csv')
        
        # --- Khởi tạo file CSV ---
        self.file_exists = os.path.exists(self.csv_path)
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Header: timestamp_ns, topic, val1 (steer_angle hoặc linear_x), val2 (speed hoặc angular_z)
        self.csv_writer.writerow(['timestamp_ns', 'topic', 'val1', 'val2'])
        self.csv_file.flush()
        
        # --- Subscribers ---
        # Lắng nghe lệnh lái được phát đi
        self.drive_sub = self.create_subscription(
            AckermannDriveStamped, 
            '/drive', 
            self.drive_callback, 
            10
        )
        # Lắng nghe phản hồi thực tế từ Odom của xe
        self.odom_sub = self.create_subscription(
            Odometry, 
            '/odom', 
            self.odom_callback, 
            10
        )
        
        self.start_time = self.get_clock().now().nanoseconds
        self.record_count = 0
        
        self.get_logger().info("=========================================")
        self.get_logger().info(" 📊 STEERING & ODOMETRY LOGGER STARTED")
        self.get_logger().info(f" Logging to: {self.csv_path}")
        self.get_logger().info(" Press Ctrl+C to stop logging.")
        self.get_logger().info("=========================================")

    def drive_callback(self, msg: AckermannDriveStamped):
        t_ns = self.get_clock().now().nanoseconds
        steer = msg.drive.steering_angle
        speed = msg.drive.speed
        
        # Ghi vào CSV: timestamp, topic_name, steering, speed
        self.csv_writer.writerow([t_ns, 'drive', steer, speed])
        self.record_count += 1
        
        # Thỉnh thoảng flush xuống disk và in log
        if self.record_count % 50 == 0:
            self.csv_file.flush()
            self.get_logger().info(f"Recorded {self.record_count} points. Current Cmd Steer: {steer:.3f} rad", throttle_duration_sec=1.0)

    def odom_callback(self, msg: Odometry):
        t_ns = self.get_clock().now().nanoseconds
        # Vận tốc dài thực tế (x) và vận tốc góc xoay thực tế (yaw_rate - z)
        vel_x = msg.twist.twist.linear.x
        yaw_rate = msg.twist.twist.angular.z
        
        # Ghi vào CSV: timestamp, topic_name, velocity_x, yaw_rate_z
        self.csv_writer.writerow([t_ns, 'odom', vel_x, yaw_rate])
        self.record_count += 1

    def destroy_node(self):
        self.csv_file.flush()
        self.csv_file.close()
        print(f"\n[steering_logger_node] Finished. Saved {self.record_count} lines to {self.csv_path}")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SteeringLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
