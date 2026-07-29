#!/usr/bin/env python3
"""
cbf_safety_filter_node.py
─────────────────────────
ROS 2 Node chạy lọc an toàn thời gian thực CBF-QP cho xe F1TENTH.

Subscribe:
  - /drive_raw (ackermann_msgs/AckermannDriveStamped) : Lệnh điều khiển chưa qua lọc từ AI / Teleop
  - /scan      (sensor_msgs/LaserScan)               : Dữ liệu cảm biến LiDAR

Publish:
  - /drive     (ackermann_msgs/AckermannDriveStamped) : Lệnh điều khiển an toàn đã qua lọc CBF-QP
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

try:
    from cbf_core import CBFQPSafetyFilter
except ImportError:
    try:
        from pure_pursuit_controller.cbf.cbf_core import CBFQPSafetyFilter
    except ImportError:
        from .cbf_core import CBFQPSafetyFilter


class CbfSafetyFilterNode(Node):
    def __init__(self):
        super().__init__('cbf_safety_filter_node')

        # --- 1. ROS 2 Parameters ---
        self.declare_parameter('d_min', 0.1)           # Khoảng cách an toàn tối thiểu (m)
        self.declare_parameter('gamma', 2.0)            # Hệ số CBF gain
        self.declare_parameter('v_max', 3.0)            # Tốc độ tối đa (m/s)
        self.declare_parameter('steer_max', 0.41)       # Góc lái tối đa (rad)
        self.declare_parameter('slack_weight', 1e4)     # Trọng số Slack variable
        self.declare_parameter('num_danger_rays', 15)   # Số tia LiDAR nguy hiểm nhất
        self.declare_parameter('fov_cutoff_deg', 75.0)  # Góc quét phía trước xét vật cản (+/- độ)
        self.declare_parameter('input_drive_topic', '/drive_raw')
        self.declare_parameter('output_drive_topic', '/drive')
        self.declare_parameter('scan_topic', 'scan_raw')

        self.d_min = self.get_parameter('d_min').value
        self.gamma = self.get_parameter('gamma').value
        self.v_max = self.get_parameter('v_max').value
        self.steer_max = self.get_parameter('steer_max').value
        self.slack_weight = self.get_parameter('slack_weight').value
        self.num_danger_rays = self.get_parameter('num_danger_rays').value
        self.fov_cutoff_deg = self.get_parameter('fov_cutoff_deg').value
        self.input_drive_topic = self.get_parameter('input_drive_topic').value
        self.output_drive_topic = self.get_parameter('output_drive_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value

        # --- 2. Initialize CBF Core ---
        self.cbf_filter = CBFQPSafetyFilter(
            d_min=self.d_min,
            gamma=self.gamma,
            v_max=self.v_max,
            steer_max=self.steer_max,
            slack_weight=self.slack_weight,
            num_danger_rays=self.num_danger_rays,
            fov_cutoff_deg=self.fov_cutoff_deg
        )

        # State
        self.latest_scan = None

        # --- 3. Pub / Sub ---
        self.sub_scan = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10
        )
        self.sub_drive_raw = self.create_subscription(
            AckermannDriveStamped,
            self.input_drive_topic,
            self.drive_raw_callback,
            10
        )
        self.pub_drive_safe = self.create_publisher(
            AckermannDriveStamped,
            self.output_drive_topic,
            10
        )

        self.get_logger().info("=========================================")
        self.get_logger().info(" CBF-QP SAFETY FILTER NODE STARTED")
        self.get_logger().info(f" Input Drive Topic  : {self.input_drive_topic}")
        self.get_logger().info(f" Output Drive Topic : {self.output_drive_topic}")
        self.get_logger().info(f" Min Safety Dist    : {self.d_min} m")
        self.get_logger().info(f" CBF Gamma          : {self.gamma}")
        self.get_logger().info("=========================================")

    def scan_callback(self, msg: LaserScan):
        """Lưu trữ scan LiDAR gần nhất"""
        self.latest_scan = msg

    def drive_raw_callback(self, msg: AckermannDriveStamped):
        """Nhận lệnh thô u_nominal, lọc qua CBF-QP và phát lệnh u_safe"""
        u_nom = np.array([msg.drive.speed, msg.drive.steering_angle], dtype=np.float32)

        if self.latest_scan is None:
            # Chưa nhận được LiDAR -> Forward thẳng lệnh thô
            self.publish_drive(u_nom[0], u_nom[1])
            return

        # 1. Trích xuất khoảng cách và góc của các tia LiDAR
        ranges = np.array(self.latest_scan.ranges, dtype=np.float32)
        angles = np.arange(len(ranges), dtype=np.float32) * self.latest_scan.angle_increment + self.latest_scan.angle_min

        # 2. Giải lọc an toàn qua CBF-QP
        u_safe = self.cbf_filter.filter(u_nom, ranges, angles)

        # Log khi CBF can thiệp bẻ lái/phanh khác với AI
        if abs(u_safe[0] - u_nom[0]) > 0.05 or abs(u_safe[1] - u_nom[1]) > 0.05:
            self.get_logger().warn(
                f"[CBF INTERVENTION] Raw: v={u_nom[0]:.2f}, steer={math.degrees(u_nom[1]):.1f}° | "
                f"Safe: v={u_safe[0]:.2f}, steer={math.degrees(u_safe[1]):.1f}°",
                throttle_duration_sec=0.5
            )

        # 3. Publish tin nhắn an toàn
        self.publish_drive(u_safe[0], u_safe[1])

    def publish_drive(self, speed: float, steering_angle: float):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'ego_racecar'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.pub_drive_safe.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CbfSafetyFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Shutting down CBF Safety Filter node.")
        node.publish_drive(0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
