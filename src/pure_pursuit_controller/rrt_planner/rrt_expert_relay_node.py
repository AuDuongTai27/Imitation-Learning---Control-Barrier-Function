#!/usr/bin/env python3
"""
rrt_expert_relay_node.py
────────────────────────
[NODE CHẠY TRÊN MÁY LAPTOP CỦA USER]
Node Chuyên Gia RRT* Né Vật Cản & Thu Thập Dữ Liệu DAgger CSV (Giữ nguyên 100% thuật toán RRT* từ code_chay_reak.py).

Vai trò:
  1. Chạy thuật toán RRT* né vật cản nặng nề trên Laptop (giảm tải CPU/GPU cho Jetson).
  2. Publish đề xuất lệnh bẻ lái của RRT* sang Topic Trung Gian: `/rrt_expert_drive`
  3. Lắng nghe tín hiệu status từ Jetson qua topic `/dagger_status` (Int32):
     - `0`: Jetson báo RRT* đang can thiệp CỨU NẾT ➔ KÍCH HOẠT LƯU DATA DAGGER TRÊN LAPTOP (với Pre-roll 2s).
     - `1`: Jetson báo AI đang tự lái ➔ Không ghi data.

Subscribe:
  /map              (nav_msgs/OccupancyGrid)                 — Bản đồ toàn cục
  /odom             (nav_msgs/Odometry)                      — Vị trí xe từ Jetson
  /scan             (sensor_msgs/LaserScan)                  — Dữ liệu LiDAR từ Jetson
  /dagger_status    (std_msgs/Int32)                         — Tín hiệu từ Jetson (0: RRT, 1: AI)

Publish:
  /rrt_expert_drive (ackermann_msgs/AckermannDriveStamped)  — Topic Trung Gian gửi sang Jetson
  /visualization/markers (visualization_msgs/MarkerArray)   — Trực quan hóa RViz
  /local_map_debug  (nav_msgs/OccupancyGrid)                 — Local Occupancy Grid
"""

import os
import sys
import csv
import math
import time
import threading
from collections import deque
import numpy as np
import scipy.interpolate as interpolate

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from tf2_ros import Buffer, TransformListener, TransformException

# Thêm đường dẫn rrt.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from rrt import RRTStarAlgorithm, treeNode


class RRTExpertRelayNode(Node):
    def __init__(self):
        super().__init__('rrt_expert_relay_node')

        home_dir = os.path.expanduser('~')
        default_dataset_path = os.path.join(
            home_dir,
            'f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/datasets/dagger_dataset_real.csv'
        )

        # ── 1. PARAMETERS (Giữ nguyên 100% từ code_chay_reak.py) ──
        self.declare_parameter("waypoint_path", "/home/adt/f1_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv")
        self.declare_parameter("lookahead_global", 2.0)
        self.declare_parameter("max_speed", 1.0)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("expert_drive_topic", "/rrt_expert_drive")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("wheelbase", 0.39)

        # DAgger Parameters trên Laptop
        self.declare_parameter("dataset_path", default_dataset_path)
        self.declare_parameter("pre_roll_secs", 2.0)
        self.declare_parameter("target_beams", 60)
        self.declare_parameter("max_range", 10.0)
        self.declare_parameter("buffer_size", 50)

        self.csv_path           = self.get_parameter("waypoint_path").value
        self.rrt_goal_dist      = self.get_parameter("lookahead_global").value
        self.max_speed          = self.get_parameter("max_speed").value
        self.map_frame          = self.get_parameter("map_frame").value
        self.base_frame         = self.get_parameter("base_frame").value
        self.wheelbase          = self.get_parameter("wheelbase").value
        
        scan_topic              = self.get_parameter("scan_topic").value
        expert_drive_topic      = self.get_parameter("expert_drive_topic").value

        self.dataset_path       = self.get_parameter("dataset_path").value
        self.pre_roll_secs      = self.get_parameter("pre_roll_secs").value
        self.target_beams       = self.get_parameter("target_beams").value
        self.max_range          = self.get_parameter("max_range").value
        self.buffer_size        = self.get_parameter("buffer_size").value

        # ── 2. CONFIG LOCAL MAP RRT (HÌNH TAM GIÁC NỚI LỎNG) ─────
        self.local_min_x = -0.5
        self.local_max_x = 3.0
        self.local_min_y = -1.0
        self.local_max_y = 1.0
        self.local_res   = 0.05

        self.last_speed    = 0.0
        self.last_steering = 0.0
        self.no_path_counter = 0

        # ── 3. DATA & BUFFERS ──────────────────────────────────────
        self.global_waypoints = self.load_waypoints(self.csv_path)
        self.global_map = None
        self.map_info   = None
        self.car_state  = None

        self.scan_data  = None
        self.scan_angles= None
        self.latest_preprocessed_scan = None

        self.lock = threading.Lock()
        self.current_dagger_status = 1  # 1: AI đang lái, 0: RRT* Cứu Nét

        # Ring Buffer lưu lịch sử 2.0s quá khứ
        self.history_ring_buffer = deque()
        self.dagger_buffer       = []
        self.total_saved_samples = 0

        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
        if not os.path.exists(self.dataset_path):
            self._write_header()

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── 4. PUB/SUB ─────────────────────────────────────────────
        # Publish lệnh Chuyên gia vào Topic Trung Gian
        self.expert_drive_pub = self.create_publisher(AckermannDriveStamped, expert_drive_topic, 10)
        self.viz_pub          = self.create_publisher(MarkerArray, '/visualization/markers', 10)
        self.local_map_pub    = self.create_publisher(OccupancyGrid, '/local_map_debug', 10)

        map_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST)

        self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        self.create_subscription(Int32, '/dagger_status', self.dagger_status_callback, 10)

        self.create_timer(0.1, self.control_loop)  # Tần số 10Hz

        self.get_logger().info("==================================================")
        self.get_logger().info(" 💻 RRT* EXPERT RELAY NODE (LAPTOP) STARTED")
        self.get_logger().info(f"  Expert Drive Topic : {expert_drive_topic}")
        self.get_logger().info(f"  DAgger Save Path   : {self.dataset_path}")
        self.get_logger().info(f"  Pre-Roll History   : {self.pre_roll_secs} s")
        self.get_logger().info("==================================================")

    def _write_header(self):
        with open(self.dataset_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [f'lidar_{i}' for i in range(self.target_beams)] + ['speed', 'steering_angle']
            writer.writerow(header)
        self.get_logger().info(f"Created new DAgger Dataset CSV at: {self.dataset_path}")

    def load_waypoints(self, file_path):
        points = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        points.append([float(row[0]), float(row[1])])
        self.get_logger().info(f"Loaded {len(points)} waypoints from {file_path}")
        return np.array(points)

    def map_callback(self, msg: OccupancyGrid):
        self.map_info = msg.info
        w = msg.info.width
        h = msg.info.height
        self.global_map = np.array(msg.data).reshape((h, w)).T
        self.get_logger().info("Global Map Received!")

    def scan_callback(self, msg: LaserScan):
        now = time.monotonic()
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, posinf=10.0, neginf=0.0)
        self.scan_data = ranges
        if self.scan_angles is None or len(self.scan_angles) != len(ranges):
            self.scan_angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

        # Tiền xử lý LiDAR 60 beams để lưu DAgger CSV
        preprocessed = self.preprocess_scan(msg)
        with self.lock:
            self.latest_preprocessed_scan = preprocessed
            # Lưu mẫu hiện tại vào Ring Buffer lịch sử (pre_roll_secs giây)
            if self.last_steering is not None:
                current_sample = list(preprocessed) + [self.max_speed, self.last_steering]
                self.history_ring_buffer.append((now, current_sample))
                while self.history_ring_buffer and (now - self.history_ring_buffer[0][0]) > self.pre_roll_secs:
                    self.history_ring_buffer.popleft()

    def dagger_status_callback(self, msg: Int32):
        now = time.monotonic()
        status = msg.data
        
        with self.lock:
            old_status = self.current_dagger_status
            self.current_dagger_status = status

            # 🚀 PHÁT HIỆN JETSON BÁO RRT* ĐANG CỨU NẾT (status chuyển sang 0)
            if status == 0:
                if old_status == 1:
                    # Vừa chuyển sang Cứu Nét -> Hút 2.0s lịch sử Pre-Roll quá khứ vào Buffer
                    preroll_count = 0
                    for ts, sample in self.history_ring_buffer:
                        self.dagger_buffer.append(sample)
                        preroll_count += 1
                    self.get_logger().warn(
                        f" 📥 [DAgger Pre-Roll] Jetson Overridden by RRT*! Grabbed {preroll_count} past samples ({self.pre_roll_secs}s)."
                    )
                else:
                    # Tiếp tục ghi mẫu hiện tại khi RRT* đang giữ lái
                    if self.latest_preprocessed_scan is not None:
                        current_sample = list(self.latest_preprocessed_scan) + [self.max_speed, self.last_steering]
                        self.dagger_buffer.append(current_sample)

                # Flush đĩa khi đủ buffer_size
                if len(self.dagger_buffer) >= self.buffer_size:
                    buffer_to_save = list(self.dagger_buffer)
                    self.dagger_buffer.clear()
                    threading.Thread(target=self._flush_dagger_buffer, args=(buffer_to_save,), daemon=True).start()

    def _flush_dagger_buffer(self, data_list):
        try:
            with open(self.dataset_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data_list)
            self.total_saved_samples += len(data_list)
            self.get_logger().info(f"[DAgger CSV Laptop] Saved +{len(data_list)} RRT* recovery samples. Total: {self.total_saved_samples}")
        except Exception as e:
            self.get_logger().error(f"Failed to flush DAgger buffer: {e}")

    # ══════════════════════════════════════════════════════════════════
    #  THUẬT TOÁN RRT* CHUYÊN GIA (CHÍNH XÁC 100% CODE_CHAY_REAK)
    # ══════════════════════════════════════════════════════════════════

    def control_loop(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            q = transform.transform.rotation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            self.car_state = [x, y, yaw]
        except TransformException:
            return

        if len(self.global_waypoints) == 0 or self.global_map is None:
            return
        cx, cy, cyaw = self.car_state

        # 1. Goal Logic
        local_goal_world = self.get_global_lookahead(cx, cy, self.rrt_goal_dist)
        dx = local_goal_world[0] - cx
        dy = local_goal_world[1] - cy
        gx_local = dx * math.cos(-cyaw) - dy * math.sin(-cyaw)
        gy_local = dx * math.sin(-cyaw) + dy * math.cos(-cyaw)

        margin = 0.2
        gx_local = np.clip(gx_local, self.local_min_x + margin, self.local_max_x - margin)
        gy_local = np.clip(gy_local, self.local_min_y + margin, self.local_max_y - margin)

        # 2. Local Map
        local_grid = self.extract_local_map(cx, cy, cyaw)
        grid_w = int((self.local_max_x - self.local_min_x) / self.local_res)
        grid_h = int((self.local_max_y - self.local_min_y) / self.local_res)

        start_idx = self.local_to_grid(0, 0)
        goal_idx  = self.local_to_grid(gx_local, gy_local)

        iter_count = 200
        if self.no_path_counter > 2:
            iter_count = 500

        rrt = RRTStarAlgorithm(
            start=start_idx, goal=goal_idx,
            interations=iter_count,
            collision_margin=1,
            steer_length=3,
            goal_tolerance=5,
            grid=local_grid
        )

        path_local_grid = []
        found = False

        # Soft Start Check
        actual_start_idx = list(start_idx)
        start_ok = False

        if local_grid[start_idx[0], start_idx[1]] != 100:
            start_ok = True
        else:
            search_rad = 5
            for dx_i in range(-search_rad, search_rad + 1):
                for dy_i in range(-search_rad, search_rad + 1):
                    nx, ny = start_idx[0] + dx_i, start_idx[1] + dy_i
                    if 0 <= nx < grid_w and 0 <= ny < grid_h:
                        if local_grid[nx, ny] != 100:
                            actual_start_idx = [nx, ny]
                            rrt.start = treeNode(nx, ny)
                            start_ok = True
                            break
                if start_ok:
                    break

        if not start_ok:
            self.get_logger().warn("CRITICAL: CAR IS BURIED IN OBSTACLES (Start Blocked)", throttle_duration_sec=2.0)
            found = False
        else:
            for i in range(rrt.iterations):
                sampled = rrt.sample()
                if sampled is None:
                    continue
                nearest_idx = rrt.nearest(rrt.tree, sampled)
                new_node = rrt.steer(rrt.tree[nearest_idx], sampled)
                if not rrt.check_collision(rrt.tree[nearest_idx], new_node):
                    new_node.parent = rrt.tree[nearest_idx]
                    rrt.tree.append(new_node)
                    if rrt.is_goal(new_node, rrt.goal_node.x, rrt.goal_node.y):
                        path_local_grid = rrt.find_path_2(new_node)
                        found = True
                        break

        # 3. Result & Calculation
        if found:
            self.no_path_counter = 0
            path_local_nodes = rrt.post_processing(path_local_grid)
            path_local_meters = []
            for n in path_local_nodes:
                lx, ly = self.grid_to_local(n.x, n.y)
                path_local_meters.append([lx, ly])
            path_local_meters.append([gx_local, gy_local])

            if actual_start_idx != list(start_idx):
                path_local_meters.insert(0, [0.0, 0.0])

            path_smooth = self.apply_b_spline(path_local_meters)
            pp_lookahead = np.clip(0.4 * self.max_speed + 0.7, 0.8, 1.5)
            target_local = path_smooth[-1]
            for pt in path_smooth:
                if math.hypot(pt[0], pt[1]) >= pp_lookahead:
                    target_local = pt
                    break

            steering = 1.5 * math.atan((2 * 0.38 * target_local[1]) / (math.hypot(target_local[0], target_local[1]) ** 2))
            self.last_speed = self.max_speed
            self.last_steering = steering

            # 🚀 PUBLISH LỆNH BẺ LÁI SANG TOPIC TRUNG GIAN /rrt_expert_drive (DÀNH CHO JETSON)
            self.publish_expert_drive(self.last_speed, steering)
            self.visualize(path_smooth, target_local)
        else:
            self.no_path_counter += 1
            if self.no_path_counter < 5:
                self.publish_expert_drive(self.last_speed, self.last_steering)
            elif self.no_path_counter < 30:
                self.publish_expert_drive(-0.8, -self.last_steering)
            else:
                self.publish_expert_drive(0.0, 0.0)

    def extract_local_map(self, cx, cy, cyaw):
        w_int = int((self.local_max_x - self.local_min_x) / self.local_res)
        h_int = int((self.local_max_y - self.local_min_y) / self.local_res)
        x_idxs = np.arange(w_int); y_idxs = np.arange(h_int)
        grid_x, grid_y = np.meshgrid(x_idxs, y_idxs, indexing='ij')

        lx = grid_x * self.local_res + self.local_min_x
        ly = grid_y * self.local_res + self.local_min_y

        local_grid = np.zeros((w_int, h_int), dtype=int)
        cos_yaw = math.cos(cyaw); sin_yaw = math.sin(cyaw)

        if self.map_info is not None:
            wx = cx + lx * cos_yaw - ly * sin_yaw
            wy = cy + lx * sin_yaw + ly * cos_yaw
            g_res = self.map_info.resolution
            g_ox  = self.map_info.origin.position.x; g_oy = self.map_info.origin.position.y
            gx = ((wx - g_ox) / g_res).astype(int)
            gy = ((wy - g_oy) / g_res).astype(int)
            valid_mask = (gx >= 0) & (gx < self.global_map.shape[0]) & (gy >= 0) & (gy < self.global_map.shape[1])
            extracted_vals = np.zeros_like(gx, dtype=int)
            extracted_vals[valid_mask] = self.global_map[gx[valid_mask], gy[valid_mask]]
            local_grid = np.where(extracted_vals == 100, 100, 0)

        if self.scan_data is not None and self.scan_angles is not None:
            mask = (self.scan_data < self.local_max_x + 1.0) & (self.scan_data > 0.05)
            valid_ranges = self.scan_data[mask]
            valid_angles = self.scan_angles[mask]
            obs_x = valid_ranges * np.cos(valid_angles)
            obs_y = valid_ranges * np.sin(valid_angles)

            in_grid_mask = (obs_x >= self.local_min_x) & (obs_x <= self.local_max_x) & \
                           (obs_y >= self.local_min_y) & (obs_y <= self.local_max_y)
            obs_x = obs_x[in_grid_mask]; obs_y = obs_y[in_grid_mask]
            obs_gx = ((obs_x - self.local_min_x) / self.local_res).astype(int)
            obs_gy = ((obs_y - self.local_min_y) / self.local_res).astype(int)
            obs_gx = np.clip(obs_gx, 0, w_int - 1); obs_gy = np.clip(obs_gy, 0, h_int - 1)

            inf_rad = int(0.35 / self.local_res)
            for i in range(len(obs_gx)):
                cx_i, cy_i = obs_gx[i], obs_gy[i]
                x_start = max(0, cx_i - inf_rad); x_end = min(w_int, cx_i + inf_rad + 1)
                y_start = max(0, cy_i - inf_rad); y_end = min(h_int, cy_i + inf_rad + 1)
                local_grid[x_start:x_end, y_start:y_end] = 100

        progress = (lx - self.local_min_x) / (self.local_max_x - self.local_min_x)
        start_width_half = 0.6
        end_width_half = self.local_max_y
        allowed_y_abs = start_width_half + progress * (end_width_half - start_width_half)
        triangle_mask = np.abs(ly) > allowed_y_abs
        local_grid[triangle_mask] = 100

        self.publish_local_grid_msg(local_grid)
        return local_grid

    def publish_local_grid_msg(self, grid_data):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg(); msg.header.frame_id = self.base_frame
        msg.info.resolution = self.local_res; msg.info.width = grid_data.shape[1]; msg.info.height = grid_data.shape[0]
        msg.info.origin.position.x = self.local_min_x; msg.info.origin.position.y = self.local_min_y
        msg.info.origin.orientation.w = 1.0
        msg.data = grid_data.T.flatten().tolist()
        self.local_map_pub.publish(msg)

    def local_to_grid(self, lx, ly):
        gx = int((lx - self.local_min_x) / self.local_res); gy = int((ly - self.local_min_y) / self.local_res)
        return gx, gy

    def grid_to_local(self, gx, gy):
        lx = gx * self.local_res + self.local_min_x; ly = gy * self.local_res + self.local_min_y
        return lx, ly

    def apply_b_spline(self, path_points):
        if len(path_points) < 3: return path_points
        try:
            x = [p[0] for p in path_points]; y = [p[1] for p in path_points]
            x_u, y_u = [x[0]], [y[0]]
            for i in range(1, len(x)):
                if math.hypot(x[i] - x[i - 1], y[i] - y[i - 1]) > 0.05:
                    x_u.append(x[i]); y_u.append(y[i])
            if len(x_u) < 3: return path_points
            tck, u = interpolate.splprep([x_u, y_u], k=3, s=0.001)
            u_fine = np.linspace(0, 1, num=50)
            x_fine, y_fine = interpolate.splev(u_fine, tck)
            return [[x_fine[i], y_fine[i]] for i in range(len(x_fine))]
        except Exception:
            return path_points

    def get_global_lookahead(self, cx, cy, dist_lookup):
        dists = np.linalg.norm(self.global_waypoints - np.array([cx, cy]), axis=1)
        nearest_idx = np.argmin(dists)
        idx = nearest_idx
        for i in range(len(self.global_waypoints)):
            idx = (nearest_idx + i) % len(self.global_waypoints)
            if math.hypot(self.global_waypoints[idx][0] - cx, self.global_waypoints[idx][1] - cy) > dist_lookup:
                return self.global_waypoints[idx]
        return self.global_waypoints[nearest_idx]

    def publish_expert_drive(self, speed, angle):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(angle)
        self.expert_drive_pub.publish(msg)

    def preprocess_scan(self, msg: LaserScan) -> np.ndarray:
        ranges     = np.array(msg.ranges, dtype=np.float32)
        crop_limit = math.radians(60.0)
        angles     = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min
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

    def visualize(self, path_local, target_local):
        ma = MarkerArray()
        del_m = Marker(); del_m.action = Marker.DELETEALL; ma.markers.append(del_m)
        path_m = Marker(); path_m.header.frame_id = self.base_frame; path_m.id = 0; path_m.type = Marker.LINE_STRIP; path_m.action = Marker.ADD
        path_m.scale.x = 0.1; path_m.color.a = 1.0; path_m.color.g = 1.0; path_m.pose.orientation.w = 1.0
        for pt in path_local:
            p = Point(); p.x = pt[0]; p.y = pt[1]; p.z = 0.2; path_m.points.append(p)
        ma.markers.append(path_m)
        tgt_m = Marker(); tgt_m.header.frame_id = self.base_frame; tgt_m.id = 1; tgt_m.type = Marker.SPHERE; tgt_m.action = Marker.ADD
        tgt_m.pose.position.x = target_local[0]; tgt_m.pose.position.y = target_local[1]; tgt_m.pose.position.z = 0.3
        tgt_m.scale.x = 0.3; tgt_m.scale.y = 0.3; tgt_m.scale.z = 0.3; tgt_m.color.a = 1.0; tgt_m.color.r = 1.0; tgt_m.color.b = 0.0
        ma.markers.append(tgt_m)
        self.viz_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = RRTExpertRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Shutting down RRT* Expert Relay Node...")
        if len(node.dagger_buffer) > 0:
            node._flush_dagger_buffer(node.dagger_buffer)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
