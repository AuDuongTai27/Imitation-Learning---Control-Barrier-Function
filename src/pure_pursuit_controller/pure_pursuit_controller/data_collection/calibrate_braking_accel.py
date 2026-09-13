#!/usr/bin/env python3
"""
calibrate_braking_accel.py
──────────────────────────
Node ROS 2 Tự Động Đo Đạc & Calibrate Gia Tốc Phanh Hãm Tối Đa (a_max_brake) Cho Xe F1TENTH.

Chức năng:
  1. Cho xe chạy thẳng tăng tốc đạt tốc độ thử nghiệm v_0 (mặc định 2.0 m/s).
  2. Gửi lệnh phanh ngắt khẩn cấp v = 0.0 m/s tại mốc thời gian xác định.
  3. Đọc dữ liệu từ Odometry (/odom) và IMU (/imu/data) để ghi nhận:
     - Thời gian phanh dừng Delta_t (s)
     - Quãng đường phanh trôi S_phanh (m)
     - Gia tốc âm cực đại từ IMU (a_imu_max)
  4. Tự động tính toán và in ra con số a_max_brake chuẩn xác nhất để nạp vào CBF!
"""

import time
import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class CalibrateBrakingAccelNode(Node):
    def __init__(self):
        super().__init__('calibrate_braking_accel_node')

        # --- 1. ROS 2 Parameters ---
        self.declare_parameter('test_speed', 2.0)         # Tốc độ chạy thử nghiệm (m/s)
        self.declare_parameter('accel_duration', 3.0)     # Thời gian chạy thẳng tăng tốc (s)
        self.declare_parameter('drive_topic', '/drive')   # Topic phát lệnh lái
        self.declare_parameter('odom_topic', '/odom')     # Topic đọc vận tốc / vị trí
        self.declare_parameter('imu_topic', '/imu/data')  # Topic đọc gia tốc IMU

        self.test_speed = self.get_parameter('test_speed').value
        self.accel_duration = self.get_parameter('accel_duration').value
        self.drive_topic = self.get_parameter('drive_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.imu_topic = self.get_parameter('imu_topic').value

        # --- 2. State Variables ---
        # 0: IDLE, 1: ACCELERATING, 2: BRAKING, 3: STOPPED & DONE
        self.state = 0
        self.start_time = None
        self.brake_start_time = None

        self.start_x = None
        self.start_y = None
        self.stop_x = None
        self.stop_y = None

        self.v_actual_start = 0.0
        self.min_imu_accel_x = 0.0
        self.imu_accel_samples = []

        self.curr_speed = 0.0
        self.curr_x = 0.0
        self.curr_y = 0.0

        # --- 3. Pub / Sub ---
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.imu_sub = self.create_subscription(Imu, self.imu_topic, self.imu_callback, 10)

        # Timer vòng lặp 50Hz (20ms)
        self.timer = self.create_timer(0.02, self.control_loop)

        self.get_logger().info("=========================================")
        self.get_logger().info(" BRAKING DECELERATION CALIBRATION NODE")
        self.get_logger().info(f" Test Speed : {self.test_speed} m/s")
        self.get_logger().info(f" Drive Topic: {self.drive_topic}")
        self.get_logger().info(f" Odom Topic : {self.odom_topic}")
        self.get_logger().info("=========================================")
        self.get_logger().info("Bat dau thu nghiem trong 2 giay nữa...")

    def odom_callback(self, msg: Odometry):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.curr_speed = math.sqrt(vx**2 + vy**2)

    def imu_callback(self, msg: Imu):
        accel_x = msg.linear_acceleration.x
        if self.state == 2:  # Đang phanh
            self.imu_accel_samples.append(accel_x)
            if accel_x < self.min_imu_accel_x:
                self.min_imu_accel_x = accel_x

    def control_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9

        if self.state == 0:
            if self.start_time is None:
                self.start_time = now
            elif now - self.start_time >= 2.0:
                self.get_logger().info(f"[PHAI CHẠY TĂNG TỐC] Tang toc len {self.test_speed} m/s...")
                self.state = 1
                self.start_time = now

        elif self.state == 1:
            # Chạy thẳng tăng tốc
            self.publish_drive(self.test_speed, 0.0)
            if now - self.start_time >= self.accel_duration:
                self.get_logger().warn("[PHANH KHẨN CẤP] PHANH NGẮT NGAY LẬP TỨC (v = 0.0 m/s)!")
                self.state = 2
                self.brake_start_time = now
                self.v_actual_start = max(0.5, self.curr_speed)
                self.start_x = self.curr_x
                self.start_y = self.curr_y
                self.min_imu_accel_x = 0.0
                self.imu_accel_samples.clear()

        elif self.state == 2:
            # Phát lệnh phanh 0.0 m/s
            self.publish_drive(0.0, 0.0)

            # Kiểm tra xem xe đã dừng hẳn chưa (vận tốc < 0.05 m/s)
            if self.curr_speed <= 0.05 and (now - self.brake_start_time) >= 0.3:
                self.state = 3
                self.stop_x = self.curr_x
                self.stop_y = self.curr_y
                dt = now - self.brake_start_time

                # Tính toán quãng đường trôi S_phanh
                dist = math.sqrt((self.stop_x - self.start_x)**2 + (self.stop_y - self.start_y)**2)

                # Tính gia tốc phanh trung bình: a = v_0 / dt
                a_avg = self.v_actual_start / dt if dt > 0 else 0.0

                # Tính gia tốc phanh theo động lực học: a = v_0^2 / (2 * dist)
                a_kinematic = (self.v_actual_start**2) / (2.0 * dist) if dist > 0 else 0.0

                # Gia tốc cực đại từ IMU (trị tuyệt đối)
                a_imu_peak = abs(self.min_imu_accel_x)

                self.get_logger().info("=========================================")
                self.get_logger().info(" 🏆 KẾT QUẢ THỬ NGHIỆM PHANH THỰC TẾ 🏆")
                self.get_logger().info(f" 1. Vận tốc bắt đầu phanh (v_0): {self.v_actual_start:.2f} m/s")
                self.get_logger().info(f" 2. Thời gian phanh dừng (dt) : {dt:.3f} giây")
                self.get_logger().info(f" 3. Quãng đường trôi phanh (S): {dist:.3f} mét ({dist*100:.1f} cm)")
                self.get_logger().info(" ─────────────────────────────────────────")
                self.get_logger().info(f" ➔ Gia tốc phanh Trung bình   : a_avg       = {a_avg:.2f} m/s^2")
                self.get_logger().info(f" ➔ Gia tốc phanh Động lực học: a_kinematic = {a_kinematic:.2f} m/s^2")
                if a_imu_peak > 0.1:
                    self.get_logger().info(f" ➔ Gia tốc phanh IMU Cực đại : a_imu_peak  = {a_imu_peak:.2f} m/s^2")
                self.get_logger().info(" ─────────────────────────────────────────")
                rec_a = (a_kinematic + a_avg) / 2.0 if a_kinematic > 0 else a_avg
                self.get_logger().info(f" 🌟 KHUYÊN DÙNG KHAI BÁO NẠP VÀO CBF: a_max_brake := {rec_a:.2f}")
                self.get_logger().info("=========================================")

        elif self.state == 3:
            # Xe dừng hẳn -> Giữ 0.0 m/s
            self.publish_drive(0.0, 0.0)

    def publish_drive(self, speed: float, steer: float):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steer)
        self.drive_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CalibrateBrakingAccelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Shutting down Braking Calibration Node.")
        node.publish_drive(0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
