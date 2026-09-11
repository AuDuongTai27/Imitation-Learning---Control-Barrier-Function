#!/usr/bin/env python3
"""
persistent_obstacle_spawner.py
───────────────────────────────
ROS 2 Node tạo và tích lũy vĩnh viễn các vật cản ảo từ sự kiện click chuột trên RViz (/clicked_point).

Hỗ trợ 3 Chế độ (Modes):
  [1] Static Mode: Vật cản tích lũy định vị CỐ ĐỊNH tại vị trí click.
  [2] Dynamic Mode (Timer): Vật cản tự động nhúc nhích / đổi vị trí xung quanh mỗi N giây.
  [3] Manual Trigger Mode (Key 'r'): Vật cản CHỈ đổi vị trí nhúc nhích xung quanh khi người dùng NHẤN PHÍM 'r' trên Terminal!

Đặc điểm:
  1. Mỗi lần bấm "Publish Point" trong RViz -> Sinh thêm 1 vật cản mới tại vị trí click.
  2. Các vật cản ĐÃ CLICK TRƯỚC ĐÓ ĐƯỢC GIỮ NGUYÊN (Tích lũy vĩnh viễn).
  3. Chèn đồng thời TOÀN BỘ danh sách vật cản vào dữ liệu LiDAR scan thô (`/scan_raw` -> `/scan`).
  4. Hiển thị MarkerArray các vật cản nổi bật trên RViz (`/sim_obstacle_array` & `/sim_obstacle`).
  5. Danh sách vật cản chỉ bị xóa hoàn toàn khi tắt Node bằng `Ctrl + C`.
"""

import sys
import math
import select
import random
import threading
import termios
import tty
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener, TransformException


class PersistentObstacleSpawnerNode(Node):
    def __init__(self):
        super().__init__('persistent_obstacle_spawner_node')

        # --- Parameters ---
        self.declare_parameter('obstacle_radius', 0.20)  # Bán kính vật cản (20cm, đường kính 40cm)
        self.declare_parameter('obstacle_height', 1.0)   # Chiều cao vật cản
        self.declare_parameter('mode', 0)                # Mode 1: Static | Mode 2: Dynamic (10s) | Mode 3: Manual Key 'r'
        self.declare_parameter('shift_interval', 20.0)   # Chu kỳ đổi vị trí ở Mode 2 (giây)
        self.declare_parameter('max_shift_offset', 0.30) # Độ lệch tối đa xung quanh vị trí gốc (mét)

        self.declare_parameter('scan_raw_topic', '/scan_raw')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('clicked_point_topic', '/clicked_point')

        self.obstacle_radius = float(self.get_parameter('obstacle_radius').value)
        self.obstacle_height = float(self.get_parameter('obstacle_height').value)
        self.mode = int(self.get_parameter('mode').value)
        self.shift_interval = float(self.get_parameter('shift_interval').value)
        self.max_shift_offset = float(self.get_parameter('max_shift_offset').value)

        scan_raw_topic = self.get_parameter('scan_raw_topic').value
        scan_topic = self.get_parameter('scan_topic').value
        clicked_point_topic = self.get_parameter('clicked_point_topic').value

        # Prompt chọn mode nếu chưa truyền param -p mode:=1/2/3
        if self.mode not in [1, 2, 3]:
            print("\n" + "=" * 65)
            print(" 🎯 CHỌN CHẾ ĐỘ THÊM VẬT CẢN (PERSISTENT OBSTACLE MODE):")
            print("  [1] - Vật cản ĐỊNH VỊ CỐ ĐỊNH (Static Accumulated)")
            print("  [2] - Vật cản ĐỘNG (Tự động đổi vị trí xung quanh mỗi N giây)")
            print("  [3] - KÍCH HOẠT THỦ CÔNG (Chỉ đổi vị trí khi NHẤN PHÍM 'r' trên Terminal)")
            print("=" * 65)
            try:
                user_in = input("👉 Nhập lựa chọn của bạn (1, 2 hoặc 3) [Mặc định: 1]: ").strip()
                if user_in == "2":
                    self.mode = 2
                elif user_in == "3":
                    self.mode = 3
                else:
                    self.mode = 1
            except Exception:
                self.mode = 1

        mode_descs = {
            1: "CỐ ĐỊNH (STATIC)",
            2: f"ĐỘNG TỰ ĐỔI VỊ TRÍ (MỖI {self.shift_interval:.0f}s)",
            3: "KÍCH HOẠT THỦ CÔNG (BẤM PHÍM 'r' TRÊN TERMINAL DỂ ĐỔI VỊ TRÍ)"
        }
        self.get_logger().info(f"👉 ĐÃ KÍCH HOẠT CHẾ ĐỘ SPAWN: [{self.mode}] - {mode_descs[self.mode]}")

        # --- TF2 Init ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Subscriptions & Publishers ---
        self.click_sub = self.create_subscription(
            PointStamped, clicked_point_topic, self.click_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, scan_raw_topic, self.scan_callback, 10
        )
        # Topic trigger đổi vị trí từ bên ngoài qua ROS 2 (nếu có)
        self.shift_sub = self.create_subscription(
            Empty, '/shift_obstacles', lambda msg: self.shift_obstacles(reason="ROS2 TOPIC TRIGGER"), 10
        )

        self.scan_pub = self.create_publisher(LaserScan, scan_topic, 10)
        self.marker_array_pub = self.create_publisher(MarkerArray, '/sim_obstacle_array', 10)
        self.marker_single_pub = self.create_publisher(Marker, '/sim_obstacle', 10)

        # Danh sách vật cản tích lũy: list của {id, origin_x, origin_y, x, y, radius}
        self.obstacles = []
        self.next_obstacle_id = 1

        # Nếu là Mode 2: Tạo Timer tự động nhúc nhích vị trí định kỳ
        if self.mode == 2:
            self.shift_timer = self.create_timer(
                self.shift_interval, lambda: self.shift_obstacles(reason=f"TIMER {self.shift_interval:.0f}s")
            )

        # Nếu là Mode 3 hoặc bất kỳ mode nào: Bật thread lắng nghe bàn phím Terminal để bắt phím 'r'
        self.keyboard_thread = threading.Thread(target=self._keyboard_listener_loop, daemon=True)
        self.keyboard_thread.start()

        self.get_logger().info("=========================================================")
        self.get_logger().info(" PERSISTENT ACCUMULATED OBSTACLE SPAWNER READY")
        self.get_logger().info(" Click 'Publish Point' in RViz toolbar to add obstacles.")
        if self.mode == 3:
            self.get_logger().info(" ⌨️  PRESS KEY 'r' ON TERMINAL TO SHIFT ALL OBSTACLES!")
        self.get_logger().info(" All clicked obstacles accumulate and persist until Ctrl+C!")
        self.get_logger().info("=========================================================")

    def click_callback(self, msg: PointStamped):
        x = float(msg.point.x)
        y = float(msg.point.y)
        obs_id = self.next_obstacle_id
        self.next_obstacle_id += 1

        # Thêm vật cản mới vào danh sách tích lũy vĩnh viễn (gốc + hiện tại)
        self.obstacles.append({
            'id': obs_id,
            'origin_x': x,
            'origin_y': y,
            'x': x,
            'y': y,
            'radius': self.obstacle_radius
        })

        self.get_logger().info(
            f"➕ Added Obstacle #{obs_id} at (x={x:.2f}, y={y:.2f}) | Total: {len(self.obstacles)}"
        )

        # Publish cập nhật MarkerArray lên RViz
        self.publish_obstacle_markers()

    def shift_obstacles(self, reason="MANUAL KEY 'r'"):
        """Dịch chuyển tất cả vật cản tới vị trí mới ngẫu nhiên xung quanh vị trí gốc"""
        if len(self.obstacles) == 0:
            self.get_logger().warn(f"[{reason}] No obstacles to shift. Click 'Publish Point' in RViz first!")
            return

        for obs in self.obstacles:
            shift_x = random.uniform(-self.max_shift_offset, self.max_shift_offset)
            shift_y = random.uniform(-self.max_shift_offset, self.max_shift_offset)
            
            obs['x'] = obs['origin_x'] + shift_x
            obs['y'] = obs['origin_y'] + shift_y

        self.get_logger().info(
            f"🔄 [{reason}] Updated positions for {len(self.obstacles)} obstacles near their origin points."
        )
        self.publish_obstacle_markers()

    def _keyboard_listener_loop(self):
        """Thread chạy ngầm lắng nghe phím 'r' từ Terminal"""
        fd = sys.stdin.fileno()
        try:
            old_settings = termios.tcgetattr(fd)
        except Exception:
            return  # Không chạy trên non-tty input

        try:
            tty.setcbreak(fd)
            while rclpy.ok():
                rlist, _, _ = select.select([sys.stdin], [], [], 0.2)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch.lower() == 'r':
                        self.shift_obstacles(reason="KEYBOARD PRESS 'r'")
        except Exception:
            pass
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass

    def publish_obstacle_markers(self):
        if len(self.obstacles) == 0:
            return

        marker_array = MarkerArray()

        for obs in self.obstacles:
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "persistent_obstacles"
            m.id = obs['id']
            m.type = Marker.CYLINDER
            m.action = Marker.ADD

            m.pose.position.x = obs['x']
            m.pose.position.y = obs['y']
            m.pose.position.z = self.obstacle_height / 2.0
            m.pose.orientation.w = 1.0

            m.scale.x = obs['radius'] * 2.0
            m.scale.y = obs['radius'] * 2.0
            m.scale.z = self.obstacle_height

            # Màu sắc theo Mode
            if self.mode == 1:
                m.color.r, m.color.g, m.color.b = 1.0, 0.2, 0.0  # Đỏ
            elif self.mode == 2:
                m.color.r, m.color.g, m.color.b = 1.0, 0.5, 0.0  # Cam
            else:
                m.color.r, m.color.g, m.color.b = 0.9, 0.1, 0.9  # Tím nổi bật cho Mode 3

            m.color.a = 1.0
            marker_array.markers.append(m)
            self.marker_single_pub.publish(m)

        self.marker_array_pub.publish(marker_array)

    def scan_callback(self, msg: LaserScan):
        if len(self.obstacles) == 0:
            self.scan_pub.publish(msg)
            return

        modified_scan = msg
        ranges = np.array(msg.ranges, dtype=np.float32)
        angles = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min

        try:
            tf = self.tf_buffer.lookup_transform(
                msg.header.frame_id,
                "map",
                rclpy.time.Time()
            )

            tx = tf.transform.translation.x
            ty = tf.transform.translation.y

            q = tf.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)

            cos_angles = np.cos(angles)
            sin_angles = np.sin(angles)

            for obs in self.obstacles:
                ox_map = obs['x']
                oy_map = obs['y']
                r_obs = obs['radius']

                ox_local = math.cos(yaw) * ox_map - math.sin(yaw) * oy_map + tx
                oy_local = math.sin(yaw) * ox_map + math.cos(yaw) * oy_map + ty

                b = -2.0 * (ox_local * cos_angles + oy_local * sin_angles)
                c = ox_local**2 + oy_local**2 - r_obs**2
                discriminant = b**2 - 4.0 * c

                valid_mask = discriminant >= 0
                if np.any(valid_mask):
                    sqrt_disc = np.sqrt(discriminant[valid_mask])
                    t1 = (-b[valid_mask] - sqrt_disc) / 2.0
                    t2 = (-b[valid_mask] + sqrt_disc) / 2.0
                    t_intersection = np.minimum(t1, t2)

                    orig_r = ranges[valid_mask]
                    update_mask = (t_intersection > 0.05) & (t_intersection < orig_r)
                    ranges[valid_mask] = np.where(update_mask, t_intersection, orig_r)

            modified_scan.ranges = ranges.tolist()

        except TransformException:
            pass

        self.scan_pub.publish(modified_scan)
        self.publish_obstacle_markers()


def main(args=None):
    rclpy.init(args=args)
    node = PersistentObstacleSpawnerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Persistent Obstacle Spawner Node — cleared all obstacles.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


