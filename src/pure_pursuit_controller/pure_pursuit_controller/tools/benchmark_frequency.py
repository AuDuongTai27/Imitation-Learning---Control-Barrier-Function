#!/usr/bin/env python3
"""
benchmark_frequency.py
──────────────────────
Node đo TẦN SỐ THỰC TẾ (Hz) của:
  - AI Model Inference  : topic /drive_raw  (output của ai_inference_real_pytorch)
  - CBF Safety Filter   : topic /drive      (output của cbf_safety_filter_node)

Sử dụng:
  ros2 run pure_pursuit_controller benchmark_frequency

Output:
  - In ra terminal mỗi N giây: Hz trung bình, min, max, latency AI→CBF
  - Lưu CSV: benchmark_frequency_<timestamp>.csv
"""

import os
import csv
import time
import math
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped

REPORT_INTERVAL_SEC = 3.0   # In báo cáo mỗi 3 giây
CSV_DIR = os.path.expanduser("~/f1_ws/benchmark_results")


class FrequencyBenchmarkNode(Node):
    def __init__(self):
        super().__init__('benchmark_frequency_node')

        # ── Timestamps buffer ──
        self._ai_times   = []   # thời điểm nhận /drive_raw (AI output)
        self._cbf_times  = []   # thời điểm nhận /drive     (CBF output)
        self._latencies  = []   # latency giữa /drive_raw và /drive kế tiếp

        self._last_ai_time  = None
        self._last_cbf_time = None

        # ── Subscribers ──
        self.create_subscription(AckermannDriveStamped, '/drive_raw',
                                 self._cb_ai,  10)
        self.create_subscription(AckermannDriveStamped, '/drive',
                                 self._cb_cbf, 10)

        # ── Timer báo cáo ──
        self.create_timer(REPORT_INTERVAL_SEC, self._report)

        # ── CSV setup ──
        os.makedirs(CSV_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(CSV_DIR, f"benchmark_frequency_{ts}.csv")
        self._csv_file = open(csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "wall_time_s",
            "ai_hz_avg", "ai_hz_min", "ai_hz_max", "ai_samples",
            "cbf_hz_avg", "cbf_hz_min", "cbf_hz_max", "cbf_samples",
            "latency_ai2cbf_ms_avg"
        ])

        self.get_logger().info("=============================================")
        self.get_logger().info(" FREQUENCY BENCHMARK NODE STARTED")
        self.get_logger().info(f" Subscribing: /drive_raw (AI) | /drive (CBF)")
        self.get_logger().info(f" Report interval: {REPORT_INTERVAL_SEC}s")
        self.get_logger().info(f" CSV output: {csv_path}")
        self.get_logger().info("=============================================")

    # ─────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────

    def _cb_ai(self, msg: AckermannDriveStamped):
        now = time.monotonic()
        self._ai_times.append(now)
        self._last_ai_time = now

    def _cb_cbf(self, msg: AckermannDriveStamped):
        now = time.monotonic()
        self._cbf_times.append(now)
        self._last_cbf_time = now

        # Tính latency AI → CBF (chỉ khi vừa nhận AI gần đây trong 100ms)
        if self._last_ai_time is not None:
            dt = (now - self._last_ai_time) * 1000.0  # ms
            if 0.0 < dt < 200.0:                       # loại bỏ outlier > 200ms
                self._latencies.append(dt)

    # ─────────────────────────────────────────────────────
    # Report
    # ─────────────────────────────────────────────────────

    def _compute_hz(self, times):
        """Tính Hz trung bình, min, max từ danh sách timestamps."""
        if len(times) < 2:
            return None, None, None
        diffs = [times[i] - times[i-1] for i in range(1, len(times))]
        hz_list = [1.0 / d for d in diffs if d > 0]
        if not hz_list:
            return None, None, None
        return (sum(hz_list) / len(hz_list),
                min(hz_list),
                max(hz_list))

    def _report(self):
        wall = time.monotonic()

        ai_avg, ai_min, ai_max = self._compute_hz(self._ai_times)
        cbf_avg, cbf_min, cbf_max = self._compute_hz(self._cbf_times)
        lat_avg = (sum(self._latencies) / len(self._latencies)
                   if self._latencies else None)

        def fmt(v, unit="Hz", dec=1):
            return f"{v:.{dec}f} {unit}" if v is not None else "N/A"

        self.get_logger().info(
            f"\n"
            f"{'='*50}\n"
            f"  [AI  /drive_raw ] avg={fmt(ai_avg)}  "
            f"min={fmt(ai_min)}  max={fmt(ai_max)}  "
            f"n={len(self._ai_times)}\n"
            f"  [CBF /drive     ] avg={fmt(cbf_avg)}  "
            f"min={fmt(cbf_min)}  max={fmt(cbf_max)}  "
            f"n={len(self._cbf_times)}\n"
            f"  [Latency AI→CBF ] avg={fmt(lat_avg, 'ms', 2)}  "
            f"samples={len(self._latencies)}\n"
            f"{'='*50}"
        )

        # Ghi CSV
        self._csv_writer.writerow([
            f"{wall:.3f}",
            f"{ai_avg:.2f}"  if ai_avg  is not None else "",
            f"{ai_min:.2f}"  if ai_min  is not None else "",
            f"{ai_max:.2f}"  if ai_max  is not None else "",
            len(self._ai_times),
            f"{cbf_avg:.2f}" if cbf_avg is not None else "",
            f"{cbf_min:.2f}" if cbf_min is not None else "",
            f"{cbf_max:.2f}" if cbf_max is not None else "",
            len(self._cbf_times),
            f"{lat_avg:.2f}" if lat_avg is not None else "",
        ])
        self._csv_file.flush()

        # Reset buffers (chỉ giữ window gần nhất)
        self._ai_times  = self._ai_times[-200:]
        self._cbf_times = self._cbf_times[-200:]
        self._latencies = self._latencies[-200:]

    def destroy_node(self):
        self._csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FrequencyBenchmarkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Benchmark stopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
