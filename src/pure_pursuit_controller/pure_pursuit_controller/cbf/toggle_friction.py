#!/usr/bin/env python3
"""
Interactive Friction Teleop for F1TENTH Gym ROS
Cho phép chuyển đổi ma sát mặt đường (Tire Friction Mu) trong thời gian thực bằng bàn phím.
"""

import sys
import select
import termios
import tty
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

HELP_TEXT = """
════════════════════════════════════════════════════════════════════
  🏎️  F1TENTH REAL-TIME ROAD FRICTION CONTROLLER (TELEOP)
════════════════════════════════════════════════════════════════════
  Phím bấm:
    [j] : CHUYỂN ĐỔI qua lại giữa Băng trơn (0.10) <-> Đường khô (1.05)
    [1] : Đặt đường khô bám chuẩn (Dry Asphalt,      μ = 1.0489)
    [2] : Đặt đường ướt trơn vừa   (Wet Track,        μ = 0.4500)
    [3] : Đặt mặt băng trượt gắt  (Black Ice / Drift, μ = 0.1000)
    [q] : Thoát chương trình
════════════════════════════════════════════════════════════════════
"""

class FrictionTeleopNode(Node):
    def __init__(self):
        super().__init__('friction_teleop_node')
        self.pub = self.create_publisher(Float32, '/set_friction', 10)
        self.current_mu = 1.0489
        self.get_logger().info("Friction Teleop Node Ready.")

    def set_friction(self, mu: float, desc: str):
        self.current_mu = mu
        msg = Float32()
        msg.data = float(mu)
        self.pub.publish(msg)
        print(f"\r>>> 🔄 ĐÃ CHUYỂN ĐỘ BÁM ĐƯỜNG SANG: μ = {mu:.4f}  [{desc}]")

    def toggle(self):
        if self.current_mu >= 0.5:
            self.set_friction(0.10, "🧊 BĂNG TRƠN TRƯỢT (BLACK ICE)")
        else:
            self.set_friction(1.0489, "🏎️  MẶT ĐƯỜNG KHÔ BÁM (DRY ASPHALT)")


def main():
    rclpy.init()
    node = FrictionTeleopNode()
    print(HELP_TEXT)
    print(f"Trạng thái ban đầu: μ = {node.current_mu:.4f} [🏎️  Đường khô bám chuẩn]")
    print("Sẵn sàng! Nhấn phím để điều khiển:")

    if not sys.stdin.isatty():
        print("Lỗi: Cần chạy trong terminal tương tác (TTY) để bắt phím bấm.")
        return

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if rlist:
                ch = sys.stdin.read(1)
                if ch in ('j', 'J'):
                    node.toggle()
                elif ch == '1':
                    node.set_friction(1.0489, "🏎️  MẶT ĐƯỜNG KHÔ BÁM (DRY ASPHALT)")
                elif ch == '2':
                    node.set_friction(0.4500, "🌧️  MẶT ĐƯỜNG ƯỚT (WET TRACK)")
                elif ch == '3':
                    node.set_friction(0.1000, "🧊 BĂNG TRƠN TRƯỢT (BLACK ICE)")
                elif ch in ('q', 'Q', '\x03'):  # q or Ctrl+C
                    print("\nThoát chương trình điều khiển ma sát.")
                    break
            rclpy.spin_once(node, timeout_sec=0.01)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
