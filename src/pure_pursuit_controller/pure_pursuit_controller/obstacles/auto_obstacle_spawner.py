#!/usr/bin/env python3
"""
auto_obstacle_spawner.py
────────────────────────
Node tự động spawn và dịch chuyển obstacle dọc theo track để phục vụ DAgger training.

Chiến lược: Track-Guided Perpendicular Spawning
  - Tìm waypoint phía trước xe N mét (spawn_distance_ahead)
  - Lệch vuông góc với hướng track ±offset ngẫu nhiên (max_perpendicular_offset)
  → Đảm bảo obstacle: nằm trên track (xe phải gặp) + không chặn hoàn toàn (RRT tìm được đường vòng)

Pipeline:
  /scan_raw  →  [LiDAR injection: chèn obstacle ảo vào tia quét]  →  /scan
  /ego_racecar/odom  →  [tính vị trí spawn theo track]
  →  /sim_obstacle (Marker RViz)

══════════════════════════════════════════════════════
  THAM SỐ CẦN TUNING (xem declare_parameter bên dưới)
══════════════════════════════════════════════════════
"""

import rclpy
from rclpy.node import Node
import math
import csv
import os
import random
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, TransformListener, TransformException


class AutoObstacleSpawnerNode(Node):
    def __init__(self):
        super().__init__('auto_obstacle_spawner_node')

        # ── Waypoint ─────────────────────────────────────────────────────
        if os.path.exists('/sim_ws'):
            default_wp = ('/sim_ws/install/waypoint/share/waypoint/'
                          'f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv')
        else:
            home = os.path.expanduser('~')
            default_wp = os.path.join(
                home, 'f1_ws/install/waypoint/share/waypoint/'
                'f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv')

        self.declare_parameter('waypoint_path', default_wp)

        # ── [TUNING] Khoảng cách spawn phía trước xe (m) ─────────────────
        # Quá gần (< 2m): xe không kịp phản ứng, data kém chất lượng
        # Quá xa (> 6m): obstacle có thể bị pass qua trước khi xe đến gần
        # → Khuyến nghị: 3.5–5.0m. Bắt đầu với 4.0m.
        self.declare_parameter('spawn_distance_ahead', 4.0)

        # ── [TUNING] Độ lệch ngang tối đa so với tim đường (m) ───────────
        self.declare_parameter('max_perpendicular_offset', 0.32)

        # ── [TUNING] Bán kính obstacle (m) ───────────────────────────────
        self.declare_parameter('obstacle_radius', 0.20)

        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('scan_raw_topic', '/scan_raw')

        # ── Đọc params ───────────────────────────────────────────────────
        self.waypoints     = self._load_waypoints(self.get_parameter('waypoint_path').value)
        self.spawn_dist    = self.get_parameter('spawn_distance_ahead').value
        self.max_offset    = self.get_parameter('max_perpendicular_offset').value
        self.obs_radius    = self.get_parameter('obstacle_radius').value
        odom_topic         = self.get_parameter('odom_topic').value
        scan_raw_topic     = self.get_parameter('scan_raw_topic').value

        # ── State ────────────────────────────────────────────────────────
        self.car_x = 0.0
        self.car_y = 0.0
        self.obs_circles = []  # List của các (x, y, radius) biểu diễn vật cản (dạng trụ đơn hoặc bức tường)
        self.obs_type_name = "CYLINDER"
        self.obs_active = False
        self._initial_spawned = False
        self.has_approached_obs = False  # Đánh dấu đã tiếp cận gần obstacle chưa

        # ── TF ───────────────────────────────────────────────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Pub / Sub ────────────────────────────────────────────────────
        self.marker_pub = self.create_publisher(Marker, '/sim_obstacle', 10)
        # 🚀 CHÚ Ý: Sub từ /scan_raw và Publish ra /scan để các node tự lái nhìn thấy được!
        self.scan_pub   = self.create_publisher(LaserScan, '/scan', 10)

        self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self.create_subscription(LaserScan, scan_raw_topic, self.scan_callback, 10)

        # Timer kiểm tra và spawn lần đầu sau 2s (để odom kịp đến)
        self.create_timer(2.0, self._try_initial_spawn)

        self._log_startup()

    # ═══════════════════════════════════════════════════════════════════
    #  CALLBACKS
    # ═══════════════════════════════════════════════════════════════════

    def odom_callback(self, msg: Odometry):
        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y

        # 🚀 CẢI TIẾN LOGIC CHUẨN: Chỉ Respawn SAU KHẢ NĂNG NÉ HOÀN TẤT
        if self.obs_active and len(self.obs_circles) > 0:
            # Khoảng cách tới tâm nhóm vật cản
            center_x, center_y, _ = self.obs_circles[0]
            dist_to_obs = math.hypot(self.car_x - center_x, self.car_y - center_y)
            
            # Bước 1: Khi xe đến gần obstacle (< 1.8m) -> Đánh dấu xe đã bắt đầu né/tiếp cận
            if dist_to_obs < 1.8:
                self.has_approached_obs = True

            # Bước 2: Chỉ khi xe đã từng đến gần VÀ giờ đã chạy xa ra khỏi obstacle (> 2.2m) -> Đã vượt qua hoàn toàn!
            if self.has_approached_obs and dist_to_obs > 2.2:
                self.get_logger().info("✅ Xe đã né và vượt qua obstacle hoàn toàn! Đang tạo obstacle mới phía trước...")
                self.has_approached_obs = False
                self._respawn_obstacle()

    def scan_callback(self, msg: LaserScan):
        """Nhận scan thô, chèn obstacle ảo (dạng trụ đơn hoặc cụm/bức tường) vào, publish lên /scan."""
        if not self.obs_active or len(self.obs_circles) == 0:
            self.scan_pub.publish(msg)
            return

        ranges = np.array(msg.ranges, dtype=np.float32)
        angles = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min

        try:
            # Tra cứu transform từ map sang laser frame chuẩn TF2
            tf = self.tf_buffer.lookup_transform(
                msg.header.frame_id, 'map', rclpy.time.Time())
            
            # Quay & Tịnh tiến chuẩn matrix 2D: P_laser = R_map_to_laser * (P_map - T_laser_in_map)
            tx = transform_tx = tf.transform.translation.x
            ty = transform_ty = tf.transform.translation.y
            q = tf.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)

            # Lặp qua tất cả các hình trụ cấu thành vật cản
            for (cx, cy, r_obs) in self.obs_circles:
                # Biến đổi tọa độ chính xác: ox_laser, oy_laser trong hệ tọa độ cảm biến LiDAR
                ox = math.cos(yaw) * cx - math.sin(yaw) * cy + tx
                oy = math.sin(yaw) * cx + math.cos(yaw) * cy + ty

                # Giao cắt tia Laser – Hình tròn
                cos_a = np.cos(angles)
                sin_a = np.sin(angles)
                b     = -2.0 * (ox * cos_a + oy * sin_a)
                c     = ox**2 + oy**2 - r_obs**2
                disc  = b**2 - 4.0 * c

                mask = disc >= 0
                if np.any(mask):
                    sq     = np.sqrt(disc[mask])
                    t1     = (-b[mask] - sq) / 2.0
                    t2     = (-b[mask] + sq) / 2.0
                    t_hit  = np.minimum(t1, t2)
                    orig   = ranges[mask]
                    
                    # Chỉ cập nhật các tia cắt trúng phía trước laser (t_hit > 0.01m)
                    update_mask = (t_hit > 0.01) & (t_hit < orig)
                    ranges[mask] = np.where(update_mask, t_hit, orig)

        except TransformException:
            pass

        out        = msg
        out.ranges = ranges.tolist()
        self.scan_pub.publish(out)

    # ═══════════════════════════════════════════════════════════════════
    #  SPAWN LOGIC (ĐÃ NÂNG CẤP DỄ HƠN DỄ NÉ HƠN)
    # ═══════════════════════════════════════════════════════════════════

    def _try_initial_spawn(self):
        if not self._initial_spawned:
            self._respawn_obstacle()
            self._initial_spawned = True

    def _respawn_obstacle(self):
        if len(self.waypoints) == 0:
            return

        n = len(self.waypoints)

        # 1. Tìm waypoint gần nhất với xe
        dists      = np.linalg.norm(self.waypoints - np.array([self.car_x, self.car_y]), axis=1)
        nearest_idx = int(np.argmin(dists))

        # 2. Random khoảng cách xa phía trước (từ 5.0m đến 8.0m)
        target_spawn_dist = random.uniform(5.0, 8.0)

        # Đi dọc theo waypoints cho đến khi tích lũy đủ target_spawn_dist
        idx      = nearest_idx
        accum    = 0.0
        for _ in range(n):
            nxt  = (idx + 1) % n
            accum += math.dist(self.waypoints[idx], self.waypoints[nxt])
            idx   = nxt
            if accum >= target_spawn_dist:
                break

        # 3. Tính hướng track (dx, dy) và vector vuông góc (perp_x, perp_y)
        wp      = self.waypoints[idx]
        wp_next = self.waypoints[(idx + 1) % n]
        dx = wp_next[0] - wp[0]
        dy = wp_next[1] - wp[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        dx /= length
        dy /= length

        # Pháp tuyến vuông góc (90° CCW)
        perp_x = -dy
        perp_y =  dx

        # 4. Đa dạng hóa hướng lệch (Trái / Giữa / Phải)
        side_choice = random.choice([-1.0, 0.0, 1.0])
        if side_choice == 0.0:
            base_offset = random.uniform(-0.05, 0.05)
        else:
            base_offset = side_choice * random.uniform(0.18, self.max_offset)

        center_x = wp[0] + perp_x * base_offset
        center_y = wp[1] + perp_y * base_offset

        # 5. 🚀 RANDOM LOẠI VẬT CẢN (ƯU TIÊN 60% TRỤ ĐƠN DỄ NÉ)
        obs_type = random.choices(
            ["SINGLE_CYLINDER", "WALL_TRANSVERSE", "WALL_LONGITUDINAL"],
            weights=[0.6, 0.2, 0.2]
        )[0]
        self.obs_circles = []

        if obs_type == "SINGLE_CYLINDER":
            self.obs_type_name = "HÌNH TRỤ ĐƠN (DỄ)"
            r = random.uniform(0.18, 0.22)  # Bán kính nhỏ 20cm
            self.obs_circles.append((center_x, center_y, r))

        elif obs_type == "WALL_TRANSVERSE":
            self.obs_type_name = "BỨC TƯỜNG NGANG VỪA"
            # Rào chắn vừa rộng ~0.5m
            r_wall = 0.12
            for step in [-0.15, 0.0, 0.15]:
                wx = center_x + perp_x * step
                wy = center_y + perp_y * step
                self.obs_circles.append((wx, wy, r_wall))

        elif obs_type == "WALL_LONGITUDINAL":
            self.obs_type_name = "BỨC TƯỜNG DỌC VỪA"
            # Rào chắn dọc rộng ~0.5m
            r_wall = 0.12
            for step in [-0.15, 0.0, 0.15]:
                wx = center_x + dx * step
                wy = center_y + dy * step
                self.obs_circles.append((wx, wy, r_wall))

        self.obs_active = True
        self._publish_marker()

        side_str = "GIỮA" if side_choice == 0 else ("TRÁI" if side_choice < 0 else "PHẢI")
        self.get_logger().info(
            f"[AUTO SPAWN] {self.obs_type_name} [{side_str}] → ({center_x:.2f}, {center_y:.2f})  "
            f"ahead={target_spawn_dist:.1f}m  offset={base_offset:+.2f}m")

    def _publish_marker(self):
        """Vẽ danh sách các Marker vật cản trên RViz"""
        if len(self.obs_circles) == 0:
            return

        # Nếu là 1 hình trụ đơn
        if len(self.obs_circles) == 1:
            cx, cy, r = self.obs_circles[0]
            m = Marker()
            m.header.frame_id     = 'map'
            m.header.stamp        = self.get_clock().now().to_msg()
            m.ns                  = 'auto_obstacles'
            m.id                  = 1
            m.type                = Marker.CYLINDER
            m.action              = Marker.ADD
            m.pose.position.x     = float(cx)
            m.pose.position.y     = float(cy)
            m.pose.position.z     = 0.5
            m.pose.orientation.w  = 1.0
            m.scale.x = m.scale.y = r * 2.0
            m.scale.z             = 1.0
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.4, 0.0, 1.0
            self.marker_pub.publish(m)
        else:
            # Nếu là bức tường (nhiều khối nối liền), vẽ dạng CYLINDER cho từng khối
            for idx, (cx, cy, r) in enumerate(self.obs_circles):
                m = Marker()
                m.header.frame_id     = 'map'
                m.header.stamp        = self.get_clock().now().to_msg()
                m.ns                  = 'auto_obstacles'
                m.id                  = idx + 1
                m.type                = Marker.CYLINDER
                m.action              = Marker.ADD
                m.pose.position.x     = float(cx)
                m.pose.position.y     = float(cy)
                m.pose.position.z     = 0.5
                m.pose.orientation.w  = 1.0
                m.scale.x = m.scale.y = r * 2.0
                m.scale.z             = 1.0
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.2, 0.2, 1.0
                self.marker_pub.publish(m)

    # ═══════════════════════════════════════════════════════════════════

    def _load_waypoints(self, path):
        if not os.path.exists(path):
            self.get_logger().error(f"Waypoint file not found: {path}")
            return np.array([])
        pts = []
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)
            for r in reader:
                pts.append([float(r[0]), float(r[1])])
        self.get_logger().info(f"Loaded {len(pts)} waypoints from {path}")
        return np.array(pts)

    def _log_startup(self):
        self.get_logger().info("=" * 50)
        self.get_logger().info("  AUTO OBSTACLE SPAWNER STARTED (EVENT-BASED)")
        self.get_logger().info(f"  Waypoints   : {len(self.waypoints)} pts")
        self.get_logger().info(f"  Spawn ahead : {self.spawn_dist} m (Random 3.0m-6.0m)")
        self.get_logger().info(f"  Max offset  : ±{self.max_offset} m (Trái / Giữa / Phải)")
        self.get_logger().info(f"  Radius      : {self.obs_radius} m")
        self.get_logger().info("=" * 50)


def main(args=None):
    rclpy.init(args=args)
    node = AutoObstacleSpawnerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
