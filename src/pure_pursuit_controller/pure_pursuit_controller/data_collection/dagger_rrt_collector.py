#!/usr/bin/env python3
"""
dagger_rrt_collector.py
───────────────────────
Node thu thập dữ liệu DAgger khi RRT expert lái xe né vật cản.

Trigger ghi data: dựa trên LiDAR nguy hiểm (khác bản DAgger cũ dùng góc lái)
  → Khi min(LiDAR trong cone ±danger_angle_deg) < danger_threshold (m):
      ghi [scan_features, expert_speed, expert_steer] vào CSV

Tại sao trigger LiDAR tốt hơn so sánh góc lái?
  - Can thiệp đúng lúc xe thực sự đối mặt obstacle
  - Ghi lại cả giai đoạn tiếp cận (xe nhìn thấy obstacle, bắt đầu lái né)
  - Không phụ thuộc vào việc AI có đang chạy hay không

Format CSV output: giống dagger_dataset_sim.csv
  [lidar_0 .. lidar_{N-1}, speed, steering_angle]

Subscribe:
  /scan  (sensor_msgs/LaserScan)                — scan đã có obstacle chèn vào
  /drive (ackermann_msgs/AckermannDriveStamped) — lệnh của RRT expert

══════════════════════════════════════════════════════
  THAM SỐ CẦN TUNING
══════════════════════════════════════════════════════
"""

import rclpy
from rclpy.node import Node
import os
import csv
import math
import time
import threading
import numpy as np

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class DAggerRRTCollectorNode(Node):
    def __init__(self):
        super().__init__('dagger_rrt_collector_node')

        # ── Output path ───────────────────────────────────────────────────
        if os.path.exists('/sim_ws'):
            default_path = ('/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/'
                            'datasets/dagger_rrt_dataset.csv')
        else:
            home = os.path.expanduser('~')
            default_path = os.path.join(
                home, 'f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/'
                'datasets/dagger_rrt_dataset.csv')

        self.declare_parameter('dataset_path', default_path)

        # ── [TUNING] Số beam LiDAR — PHẢI = input_dim khi train ──────────
        self.declare_parameter('target_beams', 60)

        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('buffer_size', 50)

        # ── [TUNING ★★★] Ngưỡng khoảng cách nguy hiểm (m) ───────────────
        # Phía trước xe trong cone ±danger_angle_deg: nếu có tia < threshold → ghi data
        #
        # Quá nhỏ (< 1.0m): chỉ ghi lúc cận kề crash → data ít, chỉ phản xạ cứu tình
        # Quá lớn (> 4.0m): ghi quá nhiều, gồm cả lúc xa obstacle → data loãng
        #
        # Gợi ý theo tốc độ:
        #   speed 1.5 m/s → threshold ≈ 1.5m
        #   speed 2.0 m/s → threshold ≈ 2.0m
        #   speed 3.0 m/s → threshold ≈ 2.5m
        # Bắt đầu với: 2.0m rồi điều chỉnh theo số samples thu được
        self.declare_parameter('danger_threshold', 2.0)

        # ── [TUNING ★★] Góc cone kiểm tra nguy hiểm (±degree) ───────────
        # 20°: chỉ quan tâm thẳng trước (bỏ qua obstacle lệch nhiều sang bên)
        # 30°: cân bằng — khuyến nghị
        # 45°: rộng, bắt cả obstacle bên cạnh (ghi nhiều hơn)
        self.declare_parameter('danger_angle_deg', 30.0)

        # ── [TUNING] Tỷ lệ ghi thêm lúc KHÔNG nguy hiểm ─────────────────
        # 0.00: chỉ ghi lúc có obstacle → model tập trung vào né tránh
        # 0.05: ghi thêm 5% sample lúc đường thông → giảm catastrophic forgetting
        # Thường để 0.0 vì model cũ (30k samples Pure Pursuit) đã có data lái thẳng
        self.declare_parameter('safe_record_rate', 0.0)

        # Timeout drive: bỏ sample nếu lệnh lái đến > X giây trước
        self.declare_parameter('drive_timeout', 0.3)

        # ── Đọc params ───────────────────────────────────────────────────
        self.dataset_path    = self.get_parameter('dataset_path').value
        self.target_beams    = self.get_parameter('target_beams').value
        self.max_range       = self.get_parameter('max_range').value
        self.buffer_size     = self.get_parameter('buffer_size').value
        self.danger_thr      = self.get_parameter('danger_threshold').value
        self.danger_angle    = self.get_parameter('danger_angle_deg').value
        self.safe_rate       = self.get_parameter('safe_record_rate').value
        self.drive_timeout   = self.get_parameter('drive_timeout').value

        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)

        # ── State ─────────────────────────────────────────────────────────
        self.latest_drive      = None
        self.latest_drive_time = 0.0
        self.buffer            = []
        self.lock              = threading.Lock()
        self.total_saved       = 0
        self.total_danger_hits = 0
        self.total_scans       = 0

        if not os.path.exists(self.dataset_path):
            self._write_header()

        # ── Pub/Sub ──────────────────────────────────────────────────────
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(AckermannDriveStamped, '/drive',
                                 self.drive_callback, 10)

        # Timer log thống kê mỗi 10s
        self.create_timer(10.0, self._log_stats)

        self._log_startup()

    # ═══════════════════════════════════════════════════════════════════
    #  CALLBACKS
    # ═══════════════════════════════════════════════════════════════════

    def drive_callback(self, msg: AckermannDriveStamped):
        with self.lock:
            self.latest_drive      = msg
            self.latest_drive_time = time.monotonic()

    def scan_callback(self, msg: LaserScan):
        now = time.monotonic()
        self.total_scans += 1

        with self.lock:
            if self.latest_drive is None or (now - self.latest_drive_time) > self.drive_timeout:
                return
            speed = self.latest_drive.drive.speed
            steer = self.latest_drive.drive.steering_angle

        # Tiền xử lý scan (giống train.py)
        scan_feat = self._preprocess_scan(msg)

        # Kiểm tra nguy hiểm
        in_danger = self._is_danger(msg)

        if in_danger:
            self.total_danger_hits += 1

        # Quyết định ghi hay không
        should_record = in_danger
        if not in_danger and self.safe_rate > 0.0 and np.random.rand() < self.safe_rate:
            should_record = True

        if not should_record:
            return

        with self.lock:
            self.buffer.append(list(scan_feat) + [speed, steer])

            if len(self.buffer) >= self.buffer_size:
                to_save = list(self.buffer)
                self.buffer.clear()
                threading.Thread(target=self._flush, args=(to_save,), daemon=True).start()

    # ═══════════════════════════════════════════════════════════════════
    #  PROCESSING
    # ═══════════════════════════════════════════════════════════════════

    def _is_danger(self, msg: LaserScan) -> bool:
        """True nếu có tia LiDAR trong cone trước ngắn hơn danger_threshold."""
        ranges    = np.array(msg.ranges, dtype=np.float32)
        angles    = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min
        crop_rad  = math.radians(self.danger_angle)
        mask      = (angles >= -crop_rad) & (angles <= crop_rad)

        if not np.any(mask):
            return False

        front = ranges[mask]
        front = np.where(np.isnan(front) | np.isinf(front), self.max_range, front)
        return float(np.min(front)) < self.danger_thr

    def _preprocess_scan(self, msg: LaserScan) -> np.ndarray:
        """Crop [-60°, +60°], downsample về target_beams — giống data_collection_sim.py."""
        ranges     = np.array(msg.ranges, dtype=np.float32)
        angles     = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min
        crop_limit = math.radians(60.0)
        mask       = (angles >= -crop_limit) & (angles <= crop_limit)

        if not np.any(mask):
            return np.full(self.target_beams, self.max_range, dtype=np.float32)

        vr = ranges[mask]
        va = angles[mask]
        vr = np.where(np.isnan(vr) | np.isinf(vr), self.max_range, vr)
        vr = np.clip(vr, 0.0, self.max_range)
        return np.interp(
            np.linspace(-crop_limit, crop_limit, self.target_beams), va, vr
        ).astype(np.float32)

    # ═══════════════════════════════════════════════════════════════════
    #  I/O
    # ═══════════════════════════════════════════════════════════════════

    def _write_header(self):
        with open(self.dataset_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [f'lidar_{i}' for i in range(self.target_beams)] + ['speed', 'steering_angle']
            writer.writerow(header)
        self.get_logger().info(f"Created DAgger RRT dataset: {self.dataset_path}")

    def _flush(self, data_list):
        try:
            with open(self.dataset_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data_list)
            self.total_saved += len(data_list)
            self.get_logger().info(
                f"[RRT COLLECTOR] +{len(data_list)} samples | "
                f"Total saved: {self.total_saved}")
        except Exception as e:
            self.get_logger().error(f"Flush error: {e}")

    def _log_stats(self):
        rate = (self.total_danger_hits / max(1, self.total_scans)) * 100
        self.get_logger().info(
            f"[STATS] Scans: {self.total_scans} | "
            f"Danger hits: {self.total_danger_hits} ({rate:.1f}%) | "
            f"Saved: {self.total_saved}")

    def _log_startup(self):
        self.get_logger().info("=" * 52)
        self.get_logger().info("  DAGGER RRT COLLECTOR STARTED")
        self.get_logger().info(f"  Output    : {self.dataset_path}")
        self.get_logger().info(f"  Beams     : {self.target_beams}")
        self.get_logger().info(f"  Danger    : < {self.danger_thr}m in ±{self.danger_angle}°")
        self.get_logger().info(f"  Safe rate : {self.safe_rate*100:.0f}%")
        self.get_logger().info("=" * 52)


def main(args=None):
    rclpy.init(args=args)
    node = DAggerRRTCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Shutting down — flushing remaining buffer...")
        with node.lock:
            remaining = list(node.buffer)
            node.buffer.clear()
        if remaining:
            node._flush(remaining)
        node.get_logger().info(
            f"Final stats: {node.total_saved} samples saved "
            f"({node.total_danger_hits} danger hits from {node.total_scans} scans)")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
