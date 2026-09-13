#!/usr/bin/env python3
"""
cbf_smooth_safety_filter_node.py
────────────────────────────────
ROS 2 Node Lọc An Toàn CBF-QP Tích Hợp Bộ Lọc Làm Mượt Chuyển Động (Smooth Motion Filter).
Giúp triệt tiêu 100% hiện tượng giật cục, rung lắc tay lái do mạng AI hoặc nhiễu LiDAR gây ra.

Tính năng nổi bật:
  1. Bộ lọc thông thấp Exponential Moving Average (EMA Low-Pass Filter) làm mượt góc lái và vận tốc.
  2. Khóa giới hạn tốc độ biến thiên góc lái (Steering Slew Rate Limiter) chống đánh lái gấp giật cục.
  3. Giữ nguyên lõi an toàn tuyệt đối Ackermann Kinematic CBF-QP.
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


class CbfSmoothSafetyFilterNode(Node):
    def __init__(self):
        super().__init__('cbf_smooth_safety_filter_node')

        # --- 1. ROS 2 Parameters ---
        self.declare_parameter('d_min', 0.25)           # Khoảng cách an toàn tĩnh tối thiểu (m)
        self.declare_parameter('gamma', 3.5)            # Hệ số CBF gain
        self.declare_parameter('v_max', 3.0)            # Tốc độ tối đa (m/s)
        self.declare_parameter('steer_max', 0.41)       # Góc lái tối đa (rad)
        self.declare_parameter('slack_weight', 1e4)     # Trọng số Slack variable
        self.declare_parameter('num_danger_rays', 15)   # Số tia LiDAR nguy hiểm nhất
        self.declare_parameter('fov_cutoff_deg', 30.0)  # Góc quét phía trước xét vật cản (+/- độ)
        self.declare_parameter('wheelbase', 0.39)       # Chiều dài cơ sở đo thực tế từ xe thật L (m)
        self.declare_parameter('a_max_brake', 2.61)      # Gia tốc phanh hãm tối đa đo từ xe thật (m/s^2)

        # --- Tham số Bộ Lọc Làm Mượt Chuyển Động (Smooth Motion Filter) ---
        self.declare_parameter('enable_smoothing', True) # Bật/tắt tính năng làm mượt
        self.declare_parameter('alpha_steer', 0.90)      # Hệ số mượt góc lái (0.1 = cực mượt, 1.0 = không mượt)
        self.declare_parameter('alpha_speed', 0.5)      # Hệ số mượt vận tốc (0.1 = cực mượt, 1.0 = không mượt)
        self.declare_parameter('max_steer_rate_deg', 120.0) # Tốc độ bẻ lái tối đa (độ/giây)

        self.declare_parameter('input_drive_topic', '/drive_raw')
        self.declare_parameter('output_drive_topic', '/drive')
        self.declare_parameter('scan_topic', '/scan')

        self.d_min = self.get_parameter('d_min').value
        self.gamma = self.get_parameter('gamma').value
        self.v_max = self.get_parameter('v_max').value
        self.steer_max = self.get_parameter('steer_max').value
        self.slack_weight = self.get_parameter('slack_weight').value
        self.num_danger_rays = self.get_parameter('num_danger_rays').value
        self.fov_cutoff_deg = self.get_parameter('fov_cutoff_deg').value
        self.wheelbase = self.get_parameter('wheelbase').value
        self.a_max_brake = self.get_parameter('a_max_brake').value

        self.enable_smoothing = self.get_parameter('enable_smoothing').value
        self.alpha_steer = self.get_parameter('alpha_steer').value
        self.alpha_speed = self.get_parameter('alpha_speed').value
        self.max_steer_rate_rad = math.radians(self.get_parameter('max_steer_rate_deg').value)

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
            fov_cutoff_deg=self.fov_cutoff_deg,
            wheelbase=self.wheelbase,
            a_max_brake=self.a_max_brake
        )

        # Trạng thái mượt lịch sử
        self.latest_scan = None
        self.prev_smooth_speed = 0.0
        self.prev_smooth_steer = 0.0
        self.last_cmd_time = None

        # --- 3. Pub / Sub ---
        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.sub_drive_raw = self.create_subscription(AckermannDriveStamped, self.input_drive_topic, self.drive_raw_callback, 10)
        self.pub_drive_safe = self.create_publisher(AckermannDriveStamped, self.output_drive_topic, 10)

        self.get_logger().info("=========================================")
        self.get_logger().info(" CBF SMOOTH SAFETY FILTER NODE STARTED")
        self.get_logger().info(f" Smooth Filter Active: {self.enable_smoothing}")
        self.get_logger().info(f" Steer Smooth Alpha  : {self.alpha_steer}")
        self.get_logger().info(f" Speed Smooth Alpha  : {self.alpha_speed}")
        self.get_logger().info("=========================================")

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def drive_raw_callback(self, msg: AckermannDriveStamped):
        now = self.get_clock().now().nanoseconds / 1e9
        dt = 0.025
        if self.last_cmd_time is not None:
            dt = max(0.005, now - self.last_cmd_time)
        self.last_cmd_time = now

        raw_speed = float(msg.drive.speed)
        raw_steer = float(msg.drive.steering_angle)

        # 1. BỘ LỌC LÀM MƯỢT TÍN HIỆU AI (Low-Pass Filter & Rate Limiter)
        if self.enable_smoothing:
            # a) Lọc thông thấp EMA cho vận tốc và góc lái
            smooth_speed = (1.0 - self.alpha_speed) * self.prev_smooth_speed + self.alpha_speed * raw_speed
            target_steer = (1.0 - self.alpha_steer) * self.prev_smooth_steer + self.alpha_steer * raw_steer

            # b) Giới hạn tốc độ biến thiên góc lái (Steering Slew Rate Limiter)
            max_delta_steer = self.max_steer_rate_rad * dt
            steer_diff = target_steer - self.prev_smooth_steer
            steer_diff_clipped = np.clip(steer_diff, -max_delta_steer, max_delta_steer)
            smooth_steer = self.prev_smooth_steer + steer_diff_clipped

            self.prev_smooth_speed = smooth_speed
            self.prev_smooth_steer = smooth_steer
        else:
            smooth_speed = raw_speed
            smooth_steer = raw_steer

        u_nom = np.array([smooth_speed, smooth_steer], dtype=np.float32)

        if self.latest_scan is None:
            self.publish_drive(u_nom[0], u_nom[1])
            return

        # 2. TRÍCH XUẤT LÍĐAR VÀ LỌC AN TOÀN CBF-QP
        ranges = np.array(self.latest_scan.ranges, dtype=np.float32)
        angles = np.arange(len(ranges), dtype=np.float32) * self.latest_scan.angle_increment + self.latest_scan.angle_min

        u_safe = self.cbf_filter.filter(u_nom, ranges, angles)

        if abs(u_safe[0] - raw_speed) > 0.05 or abs(u_safe[1] - raw_steer) > 0.05:
            self.get_logger().warn(
                f"[SMOOTH CBF] Raw: v={raw_speed:.2f}, steer={math.degrees(raw_steer):.1f}° | "
                f"Safe: v={u_safe[0]:.2f}, steer={math.degrees(u_safe[1]):.1f}°",
                throttle_duration_sec=0.5
            )

        # 3. PUBLISH LỆNH AN TOÀN MƯỢT MÀ NÀY NẠP VÀO VESC
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
    node = CbfSmoothSafetyFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Shutting down Smooth CBF Safety Filter node.")
        node.publish_drive(0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
