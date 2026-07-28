#!/usr/bin/env python3
"""
data_collection_combined_sim.py
───────────────────────────────
ROS 2 Node thu thập đồng thời dữ liệu LiDAR VÀ Odometry để huấn luyện
Imitation Learning trong môi trường mô phỏng (f1tenth_gym_ros).

Subscribe:
  - `/scan`              (sensor_msgs/msg/LaserScan)               - Dữ liệu LiDAR
  - `/ego_racecar/odom`  (nav_msgs/msg/Odometry)                   - Vị trí, hướng, vận tốc
  - `/drive`             (ackermann_msgs/msg/AckermannDriveStamped) - Lệnh lái chuyên gia

Format CSV (75 cột):
  [lidar_0 .. lidar_59,              (60 cột - LiDAR đã crop [-60°,+60°] & downsample)
   x, y, z, yaw,                     (4  cột - vị trí + góc yaw)
   qx, qy, qz, qw,                   (4  cột - quaternion)
   linear_vx, linear_vy, angular_wz, (3  cột - vận tốc)
   speed, steering_angle]            (2  cột - nhãn huấn luyện)

Chiến lược đồng bộ:
  - Mỗi khi nhận scan mới → tìm odom mới nhất (trong vòng odom_timeout=0.1s)
    và drive mới nhất (trong vòng drive_timeout=0.5s).
  - Nếu không đủ dữ liệu đồng bộ thì bỏ qua sample đó.
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
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class DataCollectionCombinedSimNode(Node):
    def __init__(self):
        super().__init__('data_collection_combined_sim_node')

        # ─── 1. Parameters ───────────────────────────────────────────────────
        if os.path.exists('/sim_ws'):
            default_dataset_path = (
                '/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/'
                'combined_dataset_sim.csv'
            )
        else:
            home_dir = os.path.expanduser('~')
            default_dataset_path = os.path.join(
                home_dir,
                'f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/'
                'combined_dataset_sim.csv'
            )

        self.declare_parameter('dataset_path',  default_dataset_path)
        self.declare_parameter('scan_topic',    '/scan')
        self.declare_parameter('odom_topic',    '/ego_racecar/odom')
        self.declare_parameter('drive_topic',   '/drive')
        self.declare_parameter('target_beams',  60)
        self.declare_parameter('max_range',     10.0)
        self.declare_parameter('buffer_size',   50)
        self.declare_parameter('odom_timeout',  0.1)   # giây – ngưỡng đồng bộ odom
        self.declare_parameter('drive_timeout', 0.5)   # giây – ngưỡng đồng bộ drive

        self.dataset_path  = self.get_parameter('dataset_path').value
        self.scan_topic    = self.get_parameter('scan_topic').value
        self.odom_topic    = self.get_parameter('odom_topic').value
        self.drive_topic   = self.get_parameter('drive_topic').value
        self.target_beams  = self.get_parameter('target_beams').value
        self.max_range     = self.get_parameter('max_range').value
        self.buffer_size   = self.get_parameter('buffer_size').value
        self.odom_timeout  = self.get_parameter('odom_timeout').value
        self.drive_timeout = self.get_parameter('drive_timeout').value

        # Tạo thư mục lưu file nếu chưa có
        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)

        # ─── 2. State & Sync Variables ────────────────────────────────────────
        self.latest_drive       = None
        self.latest_drive_time  = 0.0

        self.latest_odom        = None
        self.latest_odom_time   = 0.0

        self.buffer              = []
        self.lock                = threading.Lock()
        self.total_saved_samples = 0

        # Tạo header cho file CSV mới nếu file chưa tồn tại
        if not os.path.exists(self.dataset_path):
            self._write_header()

        # ─── 3. Subscribers ──────────────────────────────────────────────────
        self.scan_sub  = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, 10)
        self.odom_sub  = self.create_subscription(
            Odometry,  self.odom_topic, self.odom_callback, 10)
        self.drive_sub = self.create_subscription(
            AckermannDriveStamped, self.drive_topic, self.drive_callback, 10)

        self.get_logger().info("============================================")
        self.get_logger().info("  DATA COLLECTION COMBINED SIM NODE STARTED")
        self.get_logger().info(f"  Dataset   : {self.dataset_path}")
        self.get_logger().info(f"  Scan      : {self.scan_topic}  ({self.target_beams} beams)")
        self.get_logger().info(f"  Odom      : {self.odom_topic}")
        self.get_logger().info(f"  Drive     : {self.drive_topic}")
        self.get_logger().info(f"  Buffer    : {self.buffer_size} samples")
        self.get_logger().info(f"  Timeouts  : odom={self.odom_timeout}s  drive={self.drive_timeout}s")
        self.get_logger().info("============================================")

    # ─── Header ──────────────────────────────────────────────────────────────

    def _write_header(self):
        """Khởi tạo header cho file CSV kết hợp LiDAR + Odometry."""
        with open(self.dataset_path, 'w', newline='') as f:
            writer = csv.writer(f)
            lidar_cols = [f'lidar_{i}' for i in range(self.target_beams)]
            odom_cols  = [
                'x', 'y', 'z', 'yaw',
                'qx', 'qy', 'qz', 'qw',
                'linear_vx', 'linear_vy', 'angular_wz'
            ]
            label_cols = ['speed', 'steering_angle']
            writer.writerow(lidar_cols + odom_cols + label_cols)
        self.get_logger().info(
            f"Created new COMBINED dataset CSV "
            f"({self.target_beams + 13} columns): "
            f"{self.dataset_path}"
        )

    # ─── Callbacks ───────────────────────────────────────────────────────────

    def drive_callback(self, msg: AckermannDriveStamped):
        """Lưu lại lệnh điều khiển mới nhất."""
        with self.lock:
            self.latest_drive      = msg
            self.latest_drive_time = time.monotonic()

    def odom_callback(self, msg: Odometry):
        """Lưu lại odometry mới nhất (không ghi CSV ở đây)."""
        with self.lock:
            self.latest_odom      = msg
            self.latest_odom_time = time.monotonic()

    def scan_callback(self, msg: LaserScan):
        """
        Trigger chính: khi nhận scan, kiểm tra đồng bộ với odom + drive,
        ghép lại thành 1 hàng rồi đưa vào buffer.
        """
        now = time.monotonic()

        with self.lock:
            # ── Kiểm tra drive ──────────────────────────────────────────────
            if self.latest_drive is None or (now - self.latest_drive_time) > self.drive_timeout:
                return
            speed          = self.latest_drive.drive.speed
            steering_angle = self.latest_drive.drive.steering_angle

            # ── Kiểm tra odom ───────────────────────────────────────────────
            if self.latest_odom is None or (now - self.latest_odom_time) > self.odom_timeout:
                return
            odom = self.latest_odom  # snapshot an toàn (immutable ROS msg)

        # ── Tiền xử lý LiDAR (ngoài lock để không block callbacks khác) ─────
        lidar_features = self._preprocess_scan(msg)

        # ── Trích xuất Odometry ──────────────────────────────────────────────
        x  = odom.pose.pose.position.x
        y  = odom.pose.pose.position.y
        z  = odom.pose.pose.position.z

        qx = odom.pose.pose.orientation.x
        qy = odom.pose.pose.orientation.y
        qz = odom.pose.pose.orientation.z
        qw = odom.pose.pose.orientation.w

        # Tính Yaw từ Quaternion
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        linear_vx  = odom.twist.twist.linear.x
        linear_vy  = odom.twist.twist.linear.y
        angular_wz = odom.twist.twist.angular.z

        # ── Gộp thành 1 hàng và đưa vào buffer ──────────────────────────────
        row = (
            list(lidar_features)
            + [x, y, z, yaw, qx, qy, qz, qw, linear_vx, linear_vy, angular_wz]
            + [speed, steering_angle]
        )

        with self.lock:
            self.buffer.append(row)

            if len(self.buffer) >= self.buffer_size:
                buffer_to_save = list(self.buffer)
                self.buffer.clear()
                threading.Thread(
                    target=self._flush_buffer,
                    args=(buffer_to_save,),
                    daemon=True
                ).start()

    # ─── LiDAR Preprocessing ─────────────────────────────────────────────────

    def _preprocess_scan(self, msg: LaserScan) -> np.ndarray:
        """
        Crop góc quét về [-60°, +60°] phía trước và
        downsample về target_beams bằng nội suy tuyến tính.
        """
        ranges          = np.array(msg.ranges, dtype=np.float32)
        angle_min       = msg.angle_min
        angle_increment = msg.angle_increment

        crop_limit = math.radians(60.0)
        angles     = np.arange(len(ranges)) * angle_increment + angle_min

        mask = (angles >= -crop_limit) & (angles <= crop_limit)

        if not np.any(mask):
            return np.full(self.target_beams, self.max_range, dtype=np.float32)

        valid_ranges = ranges[mask]
        valid_angles = angles[mask]

        # Làm sạch NaN / Inf
        valid_ranges = np.where(
            np.isnan(valid_ranges) | np.isinf(valid_ranges),
            self.max_range,
            valid_ranges
        )
        valid_ranges = np.clip(valid_ranges, 0.0, self.max_range)

        # Nội suy về đúng target_beams điểm
        target_angles = np.linspace(-crop_limit, crop_limit, self.target_beams)
        return np.interp(target_angles, valid_angles, valid_ranges).astype(np.float32)

    # ─── Buffer Flush ─────────────────────────────────────────────────────────

    def _flush_buffer(self, data_list):
        """Ghi dữ liệu từ memory xuống file CSV (chạy trên thread phụ)."""
        try:
            with open(self.dataset_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data_list)

            self.total_saved_samples += len(data_list)
            self.get_logger().info(
                f"[COMBINED] Flushed {len(data_list)} samples. "
                f"Total: {self.total_saved_samples}"
            )
        except Exception as e:
            self.get_logger().error(f"Error flushing buffer to CSV: {e}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectionCombinedSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Shutting down combined data collection node.")
        # Flush nốt dữ liệu còn sót trong buffer trước khi dừng
        with node.lock:
            remaining = list(node.buffer)
            node.buffer.clear()
        if remaining:
            node._flush_buffer(remaining)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
