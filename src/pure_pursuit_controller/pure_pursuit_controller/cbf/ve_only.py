#!/usr/bin/env python3
"""
cbf_live_plotter.py
───────────────────
ROS 2 Node vẽ đồ thị Real-time chỉ hiển thị lệnh điều khiển thô từ AI (AI Nominal):
  1. Đồ thị Vận tốc AI Nominal Speed (m/s) theo thời gian.
  2. Đồ thị Góc bẻ lái AI Nominal Steering Angle (Độ) theo thời gian.
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
        self.declare_parameter('raw_topic', '/drive_raw')
        self.declare_parameter('window_size', 150)  # ~ 15 giây dữ liệu

        self.raw_topic = self.get_parameter('raw_topic').value
        self.window_size = self.get_parameter('window_size').value

        # --- Deque luồng dữ liệu thời gian thực ---
        self.time_hist = collections.deque(maxlen=self.window_size)
        self.v_raw_hist = collections.deque(maxlen=self.window_size)
        self.steer_raw_hist = collections.deque(maxlen=self.window_size)

        self.start_time = self.get_clock().now().nanoseconds / 1e9

        # --- Subscription ---
        self.sub_raw = self.create_subscription(
            AckermannDriveStamped,
            self.raw_topic,
            self.raw_callback,
            10
        )

        self.get_logger().info("=========================================")
        self.get_logger().info(" AI NOMINAL LIVE PLOTTER STARTED")
        self.get_logger().info(f" Subscribed Topic : {self.raw_topic}")
        self.get_logger().info("=========================================")

    def raw_callback(self, msg: AckermannDriveStamped):
        t_now = (self.get_clock().now().nanoseconds / 1e9) - self.start_time
        v_raw = msg.drive.speed
        steer_raw = math.degrees(msg.drive.steering_angle)

        # Lưu dữ liệu thời gian thực
        self.time_hist.append(t_now)
        self.v_raw_hist.append(v_raw)
        self.steer_raw_hist.append(steer_raw)


def run_plotter():
    rclpy.init()
    node = CbfLivePlotterNode()

    # Thiết lập giao diện đồ thị Matplotlib
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, (ax_v, ax_steer) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.canvas.manager.set_window_title('REAL-TIME CONTROL: AI NOMINAL SIGNALS')

    # Đồ thị 1: AI Nominal Speed
    line_v_raw, = ax_v.plot([], [], 'b-', label='AI Nominal Speed', linewidth=2.0)
    ax_v.set_ylabel('Velocity (m/s)', fontsize=11, fontweight='bold')
    ax_v.set_title('AI Nominal Velocity Command', fontsize=12, fontweight='bold', color='#0d233a')
    ax_v.legend(loc='upper right', frameon=True)
    ax_v.grid(True, linestyle='--', alpha=0.6)

    # Đồ thị 2: AI Nominal Steering Angle
    line_s_raw, = ax_steer.plot([], [], 'b-', label='AI Nominal Steering', linewidth=2.0)
    ax_steer.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax_steer.set_ylabel('Steering Angle (deg)', fontsize=11, fontweight='bold')
    ax_steer.set_title('AI Nominal Steering Signal', fontsize=12, fontweight='bold', color='#b85042')
    ax_steer.legend(loc='upper right', frameon=True)
    ax_steer.grid(True, linestyle='--', alpha=0.6)

    def update_plot(frame):
        rclpy.spin_once(node, timeout_sec=0.01)

        if len(node.time_hist) > 1:
            t = list(node.time_hist)
            vr = list(node.v_raw_hist)
            sr = list(node.steer_raw_hist)

            line_v_raw.set_data(t, vr)
            line_s_raw.set_data(t, sr)

            ax_v.set_xlim(t[0], max(t[-1], t[0] + 5.0))
            ax_v.set_ylim(-0.2, max(max(vr), 3.0) + 0.5)

            min_steer = min(min(sr), -25.0) - 5.0
            max_steer = max(max(sr), 25.0) + 5.0
            ax_steer.set_ylim(min_steer, max_steer)

        return line_v_raw, line_s_raw

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