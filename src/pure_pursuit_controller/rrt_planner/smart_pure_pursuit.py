#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math
import csv
import os
import time
import copy

# Import ROS Msgs
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, PoseStamped
from std_msgs.msg import Header

# Import RRT
from rrt import RRTStarAlgorithm, treeNode

class ContinuousLocalPlanner(Node):
    def __init__(self):
        super().__init__('continuous_local_rrt')
        
        # --- 1. CONFIG PARAMETERS ---
        self.declare_parameter("waypoint_path", "/home/adt/ros2_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv")
        self.declare_parameter("lookahead_global", 2.0) # [QUAN TRỌNG] Khoảng cách chọn đích RRT
        self.declare_parameter("max_speed", 2.0)
        
        self.csv_path = self.get_parameter("waypoint_path").value
        # Biến này tách biệt hoàn toàn với map size
        self.rrt_goal_dist = self.get_parameter("lookahead_global").value 
        self.max_speed = self.get_parameter("max_speed").value

        # --- 2. CONFIG LOCAL MAP (Vùng quét vật cản) ---
        # Hệ tọa độ xe: X tới trước, Y sang trái
        self.local_min_x = -0.5  
        self.local_max_x = 5.0   # [QUAN TRỌNG] Tầm nhìn xa (Nên lớn hơn rrt_goal_dist)
        
        # Cân đối trái phải để xe không bị lệch
        self.local_min_y = -1.2  # Sang phải 1.2m
        self.local_max_y = 1.2   # Sang trái 1.2m
        
        self.local_res = 0.05    
        
        # Biến trạng thái
        self.last_speed = 0.0
        self.last_steering = 0.0
        self.no_path_counter = 0 

        # --- DATA ---
        self.global_waypoints = self.load_waypoints(self.csv_path)
        self.global_map = None
        self.map_info = None
        self.obstacle_pos = None 
        self.last_global_idx = 0
        self.car_state = None # [x, y, yaw]
        
        # --- PUB/SUB ---
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.viz_pub = self.create_publisher(MarkerArray, '/visualization/markers', 10)
        self.local_map_pub = self.create_publisher(OccupancyGrid, '/local_map_debug', 10)

        from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
        map_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST)
        
        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        self.create_subscription(Marker, '/sim_obstacle', self.obstacle_callback, 10)

        # Timer loop 10Hz
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info(f"RRT Planner Started. Global Goal: {self.rrt_goal_dist}m. Map View: {self.local_max_x}m")

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

    def obstacle_callback(self, msg):
        if msg.action == Marker.ADD:
            self.obstacle_pos = [msg.pose.position.x, msg.pose.position.y]
        else:
            self.obstacle_pos = None

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
        self.car_state = [x, y, yaw]

    def control_loop(self):
        if self.car_state is None or len(self.global_waypoints) == 0 or self.global_map is None: return
        cx, cy, cyaw = self.car_state

        # 1. TÌM ĐIỂM ĐÍCH RRT (Dựa trên rrt_goal_dist = 2.0m)
        # Hàm này đã được sửa để nhận tham số dist_lookup
        local_goal_world = self.get_global_lookahead(cx, cy, self.rrt_goal_dist)
        
        # 2. Chuyển Goal sang Local Frame
        dx = local_goal_world[0] - cx
        dy = local_goal_world[1] - cy
        gx_local = dx * math.cos(-cyaw) - dy * math.sin(-cyaw)
        gy_local = dx * math.sin(-cyaw) + dy * math.cos(-cyaw)
        
        # Kẹp Goal vào trong Map (để RRT luôn tìm được đường)
        margin = 0.2 
        gx_local = np.clip(gx_local, self.local_min_x + margin, self.local_max_x - margin)
        gy_local = np.clip(gy_local, self.local_min_y + margin, self.local_max_y - margin)

        # 3. Tạo Local Map
        local_grid = self.extract_local_map(cx, cy, cyaw)
        
        # 4. Config RRT
        grid_w = int((self.local_max_x - self.local_min_x) / self.local_res)
        grid_h = int((self.local_max_y - self.local_min_y) / self.local_res)
        
        start_idx = self.local_to_grid(0, 0)
        goal_idx = self.local_to_grid(gx_local, gy_local)
        
        # RRT* Params (Đã tune để chạy nhanh)
        rrt = RRTStarAlgorithm(
            start=start_idx, goal=goal_idx,
            interations=150,    
            collision_margin=1, # Margin nhỏ để đi sát vật cản
            steer_length=4,     
            goal_tolerance=5,   
            grid=local_grid
        )
        
        path_local_grid = []
        found = False
        
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
        
        # 5. Xử lý kết quả
        if found:
            self.no_path_counter = 0
            path_local_nodes = rrt.post_processing(path_local_grid) 
            
            # Convert Grid -> Local Meters
            path_local_meters = []
            for n in path_local_nodes:
                lx, ly = self.grid_to_local(n.x, n.y)
                path_local_meters.append([lx, ly])
            
            # Thêm Goal vào cuối
            path_local_meters.append([gx_local, gy_local])
            
            # --- PURE PURSUIT LOGIC (Tách biệt hoàn toàn) ---
            
            # Dynamic Lookahead: Chạy nhanh nhìn xa, chạy chậm nhìn gần
            # Công thức: L = 0.4 * speed + 0.6
            pp_lookahead = np.clip(0.4 * self.max_speed + 0.6, 0.8, 1.5)
            
            target_local = path_local_meters[-1]
            for pt in path_local_meters:
                dist = math.hypot(pt[0], pt[1])
                if dist >= pp_lookahead:
                    target_local = pt
                    break
            
            # Tính góc lái
            steering = math.atan((2 * 0.33 * target_local[1]) / (math.hypot(target_local[0], target_local[1])**2))
            
            # Update state
            self.last_speed = self.max_speed
            self.last_steering = steering
            
            self.publish_drive(self.last_speed, steering)
            self.visualize(path_local_meters, target_local)
            
        else:
            self.no_path_counter += 1
            if self.no_path_counter < 5:
                self.get_logger().warn(f"Coasting... ({self.no_path_counter})")
                self.publish_drive(self.last_speed, self.last_steering)
            else:
                self.get_logger().error("Path lost! STOPPING.")
                self.publish_drive(0.0, 0.0)

    # --- HÀM HỖ TRỢ (Đã sửa hàm get_global_lookahead) ---
    def get_global_lookahead(self, cx, cy, dist_lookup):
        # Hàm này giờ nhận tham số dist_lookup từ ngoài truyền vào
        dists = np.linalg.norm(self.global_waypoints - np.array([cx, cy]), axis=1)
        nearest_idx = np.argmin(dists)
        self.last_global_idx = nearest_idx
        
        idx = nearest_idx
        for i in range(len(self.global_waypoints)):
            idx = (nearest_idx + i) % len(self.global_waypoints)
            # So sánh với tham số truyền vào
            if math.hypot(self.global_waypoints[idx][0]-cx, self.global_waypoints[idx][1]-cy) > dist_lookup:
                return self.global_waypoints[idx]
        return self.global_waypoints[nearest_idx]

    # ... (Các hàm extract_local_map, publish, visualize giữ nguyên như cũ) ...
    # Để code gọn tôi không paste lại đoạn dưới, bạn dùng đoạn dưới của phiên bản trước nhé.
    # Nhưng nhớ thay thế hàm get_global_lookahead bằng hàm mới ở trên.
    
    # --- (Copy đoạn Extract Map Numpy từ phiên bản trước vào đây) ---
    def extract_local_map(self, cx, cy, cyaw):
        w_int = int((self.local_max_x - self.local_min_x) / self.local_res)
        h_int = int((self.local_max_y - self.local_min_y) / self.local_res)
        x_idxs = np.arange(w_int); y_idxs = np.arange(h_int)
        grid_x, grid_y = np.meshgrid(x_idxs, y_idxs, indexing='ij')
        lx = grid_x * self.local_res + self.local_min_x
        ly = grid_y * self.local_res + self.local_min_y
        cos_yaw = math.cos(cyaw); sin_yaw = math.sin(cyaw)
        wx = cx + lx * cos_yaw - ly * sin_yaw
        wy = cy + lx * sin_yaw + ly * cos_yaw
        if self.map_info is None: return np.zeros((w_int, h_int))
        g_res = self.map_info.resolution; g_ox = self.map_info.origin.position.x; g_oy = self.map_info.origin.position.y
        gx = ((wx - g_ox) / g_res).astype(int); gy = ((wy - g_oy) / g_res).astype(int)
        valid_mask = (gx >= 0) & (gx < self.global_map.shape[0]) & (gy >= 0) & (gy < self.global_map.shape[1])
        local_grid = np.zeros((w_int, h_int), dtype=int)
        extracted_vals = np.zeros_like(gx, dtype=int)
        extracted_vals[valid_mask] = self.global_map[gx[valid_mask], gy[valid_mask]]
        local_grid = np.where(extracted_vals == 100, 100, 0)
        if self.obstacle_pos is not None:
            dx = self.obstacle_pos[0] - cx; dy = self.obstacle_pos[1] - cy
            ox_local = dx * cos_yaw + dy * sin_yaw; oy_local = -dx * sin_yaw + dy * cos_yaw
            dist_sq = (lx - ox_local)**2 + (ly - oy_local)**2
            obs_radius_sq = 0.3**2 # Bán kính vật cản 30cm
            local_grid[dist_sq < obs_radius_sq] = 100
        self.publish_local_grid_msg(local_grid)
        return local_grid

    def publish_local_grid_msg(self, grid_data):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg(); msg.header.frame_id = "base_link"
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

    def publish_drive(self, speed, angle):
        msg = AckermannDriveStamped()
        msg.drive.speed = float(speed); msg.drive.steering_angle = float(angle)
        self.drive_pub.publish(msg)

    def visualize(self, path_local, target_local):
        ma = MarkerArray()
        del_m = Marker(); del_m.action = Marker.DELETEALL; ma.markers.append(del_m)
        path_m = Marker(); path_m.header.frame_id = "base_link"; path_m.id = 0; path_m.type = Marker.LINE_STRIP; path_m.action = Marker.ADD
        path_m.scale.x = 0.1; path_m.color.a=1.0; path_m.color.g=1.0; path_m.pose.orientation.w = 1.0
        for pt in path_local:
            p = Point(); p.x=pt[0]; p.y=pt[1]; p.z=0.2; path_m.points.append(p)
        ma.markers.append(path_m)
        tgt_m = Marker(); tgt_m.header.frame_id = "base_link"; tgt_m.id = 1; tgt_m.type = Marker.SPHERE; tgt_m.action = Marker.ADD
        tgt_m.pose.position.x = target_local[0]; tgt_m.pose.position.y = target_local[1]; tgt_m.pose.position.z = 0.3
        tgt_m.scale.x=0.3; tgt_m.scale.y=0.3; tgt_m.scale.z=0.3; tgt_m.color.a=1.0; tgt_m.color.r=1.0; tgt_m.color.b=0.0
        ma.markers.append(tgt_m)
        self.viz_pub.publish(ma)

def main(args=None):
    rclpy.init(args=args)
    node = ContinuousLocalPlanner()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()