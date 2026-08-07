#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import numpy as np
import math
import csv
import os
import time
import copy

import scipy.interpolate as interpolate
from tf2_ros import Buffer, TransformListener, TransformException

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from rrt import RRTStarAlgorithm, treeNode

class HybridPlannerSim(Node):
    def __init__(self):
        super().__init__('hybrid_planner_sim')
        
        # ==========================================
        # 1. PARAMETERS
        # ==========================================
        self.declare_parameter("waypoint_path", "/sim_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv")
        self.declare_parameter("lookahead_global", 2.0) 
        self.declare_parameter("max_speed", 3.0) # Tốc độ tối đa trong mô phỏng (mặc định 3.0)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "ego_racecar/base_link") # Frame xe trong mô phỏng
        self.declare_parameter("wheelbase", 0.39) # Chiều dài cơ sở mô phỏng

        self.csv_path = self.get_parameter("waypoint_path").value
        self.rrt_goal_dist = self.get_parameter("lookahead_global").value
        self.max_speed = self.get_parameter("max_speed").value
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.wheelbase = self.get_parameter("wheelbase").value
        
        scan_topic = self.get_parameter("scan_topic").value
        drive_topic = self.get_parameter("drive_topic").value

        # ==========================================
        # 2. CONFIG LOCAL MAP (HÌNH TAM GIÁC RỘNG HƠN)
        # ==========================================
        self.local_min_x = -0.5  
        self.local_max_x = 3.0  
        self.local_min_y = -1.0  
        self.local_max_y = 1.0   
        self.local_res = 0.05    
        
        self.last_speed = 0.0
        self.last_steering = 0.0
        self.no_path_counter = 0 

        # ==========================================
        # 3. DATA & TF INIT
        # ==========================================
        self.global_waypoints = self.load_waypoints(self.csv_path)
        self.global_map = None
        self.map_info = None
        self.car_state = None 
        
        self.scan_data = None
        self.scan_angles = None
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # ==========================================
        # 4. PUB/SUB
        # ==========================================
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.viz_pub = self.create_publisher(MarkerArray, '/visualization/markers', 10)
        self.local_map_pub = self.create_publisher(OccupancyGrid, '/local_map_debug', 10)

        from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
        map_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST)
        
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)

        self.create_timer(0.05, self.control_loop) # Chạy ở tần số 20Hz trong mô phỏng
        
        self.get_logger().info("SIMULATION RRT (Persistent Mode) Started.")

    def load_waypoints(self, file_path):
        points = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2: points.append([float(row[0]), float(row[1])])
        return np.array(points)

    def map_callback(self, msg):
        self.map_info = msg.info
        w = msg.info.width; h = msg.info.height
        self.global_map = np.array(msg.data).reshape((h, w)).T
        self.get_logger().info("Global Map Received!")

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, posinf=10.0, neginf=0.0)
        self.scan_data = ranges
        if self.scan_angles is None or len(self.scan_angles) != len(ranges):
            self.scan_angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

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

        if len(self.global_waypoints) == 0 or self.global_map is None: return
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
        
        # 3. RRT Config
        grid_w = int((self.local_max_x - self.local_min_x) / self.local_res)
        grid_h = int((self.local_max_y - self.local_min_y) / self.local_res)
        
        start_idx = self.local_to_grid(0, 0)
        goal_idx = self.local_to_grid(gx_local, gy_local)
        
        # [TUNING] Tăng số vòng lặp lên để RRT "kiên trì" tìm lỗ hổng
        # 150 là hơi ít cho map hẹp. 350 là an toàn.
        iter_count = 200 
        if self.no_path_counter > 2: iter_count = 500 # Khi kẹt thì tìm kỹ hơn nữa

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
        
        # --- [NEW] SOFT START CHECK ---
        # Kiểm tra xem điểm Start có bị chặn không?
        # Thay vì Fail ngay, hãy tìm điểm lân cận còn trống để Start
        actual_start_idx = list(start_idx)
        start_ok = False
        
        if local_grid[start_idx[0], start_idx[1]] != 100:
            start_ok = True
        else:
            # Tìm xung quanh 5 ô (25cm) xem có chỗ nào trống không
            search_rad = 5 
            for dx in range(-search_rad, search_rad+1):
                for dy in range(-search_rad, search_rad+1):
                    nx, ny = start_idx[0] + dx, start_idx[1] + dy
                    if 0 <= nx < grid_w and 0 <= ny < grid_h:
                        if local_grid[nx, ny] != 100:
                            actual_start_idx = [nx, ny]
                            rrt.start = treeNode(nx, ny) # Cập nhật điểm start cho RRT
                            start_ok = True
                            break
                if start_ok: break
        
        if not start_ok:
            self.get_logger().warn("CRITICAL: CAR IS BURIED IN OBSTACLES (Start Blocked)")
            found = False
        else:
            # Chạy RRT bình thường
            for i in range(rrt.iterations):
                sampled = rrt.sample()
                if sampled is None: continue
                nearest_idx = rrt.nearest(rrt.tree, sampled)
                new_node = rrt.steer(rrt.tree[nearest_idx], sampled)
                if not rrt.check_collision(rrt.tree[nearest_idx], new_node):
                    new_node.parent = rrt.tree[nearest_idx]
                    rrt.tree.append(new_node)
                    if rrt.is_goal(new_node, rrt.goal_node.x, rrt.goal_node.y):
                        path_local_grid = rrt.find_path_2(new_node)
                        found = True
                        break
        
        # 5. RESULT & RECOVERY
        if found:
            self.no_path_counter = 0
            path_local_nodes = rrt.post_processing(path_local_grid) 
            path_local_meters = []
            for n in path_local_nodes:
                lx, ly = self.grid_to_local(n.x, n.y)
                path_local_meters.append([lx, ly])
            path_local_meters.append([gx_local, gy_local])
            
            # Nếu dùng Soft Start (điểm bắt đầu lệch), ta chèn thêm điểm (0,0) vào đầu
            # để đường đi bắt đầu từ tâm xe
            if actual_start_idx != list(start_idx):
                path_local_meters.insert(0, [0.0, 0.0])

            path_smooth = self.apply_b_spline(path_local_meters)
            
            pp_lookahead = np.clip(0.4 * self.max_speed + 0.7, 0.8, 1.5)
            target_local = path_smooth[-1]
            for pt in path_smooth:
                if math.hypot(pt[0], pt[1]) >= pp_lookahead:
                    target_local = pt; break
            
            steering = 1.5*math.atan((2 * self.wheelbase * target_local[1]) / (math.hypot(target_local[0], target_local[1])**2))
            
            self.last_speed = self.max_speed
            self.last_steering = steering
            self.publish_drive(self.last_speed, steering)
            self.visualize(path_smooth, target_local)
            
        else:
            self.no_path_counter += 1
            if self.no_path_counter < 5:
                self.get_logger().warn(f"Coasting... ({self.no_path_counter})")
                self.publish_drive(self.last_speed, self.last_steering)
            elif self.no_path_counter < 30: # Tăng thời gian lùi lên (3s)
                self.get_logger().error(f"STUCK! REVERSING... ({self.no_path_counter})")
                self.publish_drive(-0.8, -self.last_steering) # Đánh lái ngược để thoát góc
            else:
                self.get_logger().error("GAVE UP. STOP.")
                self.publish_drive(0.0, 0.0)

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
            g_ox = self.map_info.origin.position.x; g_oy = self.map_info.origin.position.y
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
            
            inf_rad = int(0.15 / self.local_res) 
            for i in range(len(obs_gx)):
                cx_i, cy_i = obs_gx[i], obs_gy[i]
                x_start = max(0, cx_i - inf_rad); x_end = min(w_int, cx_i + inf_rad + 1)
                y_start = max(0, cy_i - inf_rad); y_end = min(h_int, cy_i + inf_rad + 1)
                local_grid[x_start:x_end, y_start:y_end] = 100

        # --- CẮT MAP TAM GIÁC (Nới lỏng) ---
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
        # Sửa lỗi tráo đổi kích thước: grid_data.T có dạng (h_int, w_int), 
        # do đó số cột (width) phải là w_int (shape[0]) và số hàng (height) phải là h_int (shape[1])
        msg.info.resolution = self.local_res
        msg.info.width = grid_data.shape[0]   # w_int (chiều dọc x)
        msg.info.height = grid_data.shape[1]  # h_int (chiều ngang y)
        msg.info.origin.position.x = self.local_min_x
        msg.info.origin.position.y = self.local_min_y
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
                if math.hypot(x[i]-x[i-1], y[i]-y[i-1]) > 0.05:
                    x_u.append(x[i]); y_u.append(y[i])
            if len(x_u) < 3: return path_points
            tck, u = interpolate.splprep([x_u, y_u], k=3, s=0.001)
            u_fine = np.linspace(0, 1, num=50)
            x_fine, y_fine = interpolate.splev(u_fine, tck)
            smoothed_path = []
            for i in range(len(x_fine)):
                smoothed_path.append([x_fine[i], y_fine[i]])
            return smoothed_path
        except Exception:
            return path_points

    def get_global_lookahead(self, cx, cy, dist_lookup):
        dists = np.linalg.norm(self.global_waypoints - np.array([cx, cy]), axis=1)
        nearest_idx = np.argmin(dists)
        idx = nearest_idx
        for i in range(len(self.global_waypoints)):
            idx = (nearest_idx + i) % len(self.global_waypoints)
            if math.hypot(self.global_waypoints[idx][0]-cx, self.global_waypoints[idx][1]-cy) > dist_lookup:
                return self.global_waypoints[idx]
        return self.global_waypoints[nearest_idx]

    def publish_drive(self, speed, angle):
        msg = AckermannDriveStamped()
        msg.drive.speed = float(speed); msg.drive.steering_angle = float(angle)
        self.drive_pub.publish(msg)

    def visualize(self, path_local, target_local):
        ma = MarkerArray()
        del_m = Marker(); del_m.action = Marker.DELETEALL; ma.markers.append(del_m)
        path_m = Marker(); path_m.header.frame_id = self.base_frame; path_m.id = 0; path_m.type = Marker.LINE_STRIP; path_m.action = Marker.ADD
        path_m.scale.x = 0.1; path_m.color.a=1.0; path_m.color.g=1.0; path_m.pose.orientation.w = 1.0
        for pt in path_local:
            p = Point(); p.x=pt[0]; p.y=pt[1]; p.z=0.2; path_m.points.append(p)
        ma.markers.append(path_m)
        tgt_m = Marker(); tgt_m.header.frame_id = self.base_frame; tgt_m.id = 1; tgt_m.type = Marker.SPHERE; tgt_m.action = Marker.ADD
        tgt_m.pose.position.x = target_local[0]; tgt_m.pose.position.y = target_local[1]; tgt_m.pose.position.z = 0.3
        tgt_m.scale.x=0.3; tgt_m.scale.y=0.3; tgt_m.scale.z=0.3; tgt_m.color.a=1.0; tgt_m.color.r=1.0; tgt_m.color.b=0.0
        ma.markers.append(tgt_m)
        self.viz_pub.publish(ma)

def main(args=None):
    rclpy.init(args=args)
    node = HybridPlannerSim()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()