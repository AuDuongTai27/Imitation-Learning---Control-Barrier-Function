#!/usr/bin/env python3
"""
cbf_live_plotter.py
───────────────────
ROS 2 Node vẽ đồ thị Real-time chỉ hiển thị lệnh điều khiển an toàn qua CBF (CBF Safe):
  1. Đồ thị Vận tốc CBF Safe Speed (m/s) thực tế gửi xuống xe theo thời gian.
  2. Đồ thị Góc bẻ lái CBF Safe Steering Angle (Độ) thực tế gửi xuống xe theo thời gian.
"""

import math
import collections
import numpy as np

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped

import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class CbfLivePlotterNode(Node):
    def __init__(self):
        super().__init__('cbf_live_plotter_node')

        # --- Parameters ---
        self.declare_parameter('safe_topic', '/drive')
        self.declare_parameter('window_size', 150)  # ~ 15 giây dữ liệu hiển thị

        self.safe_topic = self.get_parameter('safe_topic').value
        self.window_size = self.get_parameter('window_size').value

        # --- Deque luồng dữ liệu thời gian thực ---
        self.time_hist = collections.deque(maxlen=self.window_size)
        self.v_safe_hist = collections.deque(maxlen=self.window_size)
        self.steer_safe_hist = collections.deque(maxlen=self.window_size)

        self.start_time = self.get_clock().now().nanoseconds / 1e9

        # --- Subscription ---
        self.sub_safe = self.create_subscription(
            AckermannDriveStamped,
            self.safe_topic,
            self.safe_callback,
            10
        )

        self.get_logger().info("=========================================")
        self.get_logger().info(" CBF SAFE SIGNALS LIVE PLOTTER STARTED")
        self.get_logger().info(f" Subscribed Topic : {self.safe_topic}")
        self.get_logger().info("=========================================")

    def safe_callback(self, msg: AckermannDriveStamped):
        t_now = (self.get_clock().now().nanoseconds / 1e9) - self.start_time
        v_safe = msg.drive.speed
        steer_safe = math.degrees(msg.drive.steering_angle)

        # Lưu dữ liệu thời gian thực
        self.time_hist.append(t_now)
        self.v_safe_hist.append(v_safe)
        self.steer_safe_hist.append(steer_safe)


def run_plotter():
    rclpy.init()
    node = CbfLivePlotterNode()

    # Thiết lập giao diện đồ thị Matplotlib
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, (ax_v, ax_steer) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.canvas.manager.set_window_title('REAL-TIME CONTROL: CBF SAFE COMMANDS SENT TO VEHICLE')

    # Đồ thị 1: CBF Safe Speed
    line_v_safe, = ax_v.plot([], [], 'g-', label='CBF Safe Speed (Actual command sent to vehicle)', linewidth=2.0)
    ax_v.set_ylabel('Velocity (m/s)', fontsize=11, fontweight='bold')
    ax_v.set_title('CBF Safe Velocity Command', fontsize=12, fontweight='bold', color='#0d233a')
    ax_v.legend(loc='upper right', frameon=True)
    ax_v.grid(True, linestyle='--', alpha=0.6)

    # Đồ thị 2: CBF Safe Steering Angle
    line_s_safe, = ax_steer.plot([], [], 'r-', label='CBF Safe Steering (Actual command sent to vehicle)', linewidth=2.0)
    ax_steer.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax_steer.set_ylabel('Steering Angle (deg)', fontsize=11, fontweight='bold')
    ax_steer.set_title('CBF Safe Steering Signal', fontsize=12, fontweight='bold', color='#b85042')
    ax_steer.legend(loc='upper right', frameon=True)
    ax_steer.grid(True, linestyle='--', alpha=0.6)

    def update_plot(frame):
        rclpy.spin_once(node, timeout_sec=0.01)

        if len(node.time_hist) > 1:
            t = list(node.time_hist)
            vs = list(node.v_safe_hist)
            ss = list(node.steer_safe_hist)

            line_v_safe.set_data(t, vs)
            line_s_safe.set_data(t, ss)

            ax_v.set_xlim(t[0], max(t[-1], t[0] + 5.0))
            ax_v.set_ylim(-0.2, max(max(vs), 3.0) + 0.5)

            min_steer = min(min(ss), -25.0) - 5.0
            max_steer = max(max(ss), 25.0) + 5.0
            ax_steer.set_ylim(min_steer, max_steer)

        return line_v_safe, line_s_safe

    ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    run_plotter()