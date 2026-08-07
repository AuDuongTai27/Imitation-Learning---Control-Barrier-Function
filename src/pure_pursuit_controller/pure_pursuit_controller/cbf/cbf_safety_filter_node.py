#!/usr/bin/env python3
"""
cbf_safety_filter_node.py
─────────────────────────
ROS 2 Node for real-time CBF-QP safety filtering on F1TENTH.
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

        # --- 1. Parameters ---
        self.declare_parameter('d_min', 0.1)           # Minimum safety distance (m)
        self.declare_parameter('gamma', 2.0)            # CBF gain parameter
        self.declare_parameter('v_max', 3.0)            # Max speed (m/s)
        self.declare_parameter('steer_max', 0.41)       # Max steering angle (rad)
        self.declare_parameter('slack_weight', 1e4)     # Slack weight
        self.declare_parameter('num_danger_rays', 15)   # Number of danger LiDAR rays
        self.declare_parameter('input_drive_topic', '/drive_raw')
        self.declare_parameter('output_drive_topic', '/drive')
        self.declare_parameter('scan_topic', 'scan_raw')

        self.d_min = self.get_parameter('d_min').value
        self.gamma = self.get_parameter('gamma').value
        self.v_max = self.get_parameter('v_max').value
        self.steer_max = self.get_parameter('steer_max').value
        self.slack_weight = self.get_parameter('slack_weight').value
        self.num_danger_rays = self.get_parameter('num_danger_rays').value
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
            num_danger_rays=self.num_danger_rays
        )

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
        self.latest_scan = msg

    def drive_raw_callback(self, msg: AckermannDriveStamped):
        u_nom = np.array([msg.drive.speed, msg.drive.steering_angle], dtype=np.float32)

        if self.latest_scan is None:
            self.publish_drive(u_nom[0], u_nom[1])
            return

        ranges = np.array(self.latest_scan.ranges, dtype=np.float32)
        angles = np.arange(len(ranges), dtype=np.float32) * self.latest_scan.angle_increment + self.latest_scan.angle_min

        u_safe = self.cbf_filter.filter(u_nom, ranges, angles)

        if abs(u_safe[0] - u_nom[0]) > 0.05 or abs(u_safe[1] - u_nom[1]) > 0.05:
            self.get_logger().warn(
                f"[CBF INTERVENTION] Raw: v={u_nom[0]:.2f}, steer={math.degrees(u_nom[1]):.1f}° | "
                f"Safe: v={u_safe[0]:.2f}, steer={math.degrees(u_safe[1]):.1f}°",
                throttle_duration_sec=0.5
            )

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
