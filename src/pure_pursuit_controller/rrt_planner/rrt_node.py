#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker
import numpy as np
import csv
import os
import math

# Import các class QoS cần thiết
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

from rrt import RRTStarAlgorithm, treeNode
from treeViz import TreeViz
from utils import Utils

class RRTPlannerNode(Node):
    def __init__(self):
        super().__init__('rrt_planner_node')
        
        # --- Config ---
        self.declare_parameter('csv_output_path', os.path.expanduser('~/rrt_temp_path.csv'))
        self.csv_path = self.get_parameter('csv_output_path').value
        
        self.grid = None
        self.resolution = 0.05
        self.origin = [-10.0, -10.0]
        self.viz = None
        
        # --- CẤU HÌNH QoS CHO MAP (ĐÃ KHÔI PHỤC) ---
        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        
        # --- Pub/Sub ---
        # Lắng nghe yêu cầu tìm đường từ Controller
        self.create_subscription(Path, '/trigger_replan', self.replan_callback, 10)
        
        # Đăng ký nhận Map với QoS chuẩn
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        
        # Báo cáo khi tìm đường xong
        self.success_pub = self.create_publisher(Bool, '/replan_success', 10)
        self.car_anim_pub = self.create_publisher(Marker, '/car_animation', 10)

        self.get_logger().info("RRT Service Ready. Waiting for Map & /trigger_replan...")

    def map_callback(self, msg):
        self.resolution = msg.info.resolution
        self.origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        self.width = msg.info.width
        self.height = msg.info.height
        data = np.array(msg.data).reshape((self.height, self.width))
        self.grid = data.T
        if self.viz is None:
            self.viz = TreeViz(self, self.resolution, self.origin)
            self.get_logger().info("Map Received & Viz Initialized.")

    def replan_callback(self, msg):
        if self.grid is None: 
            self.get_logger().warn("No Map received yet! Check QoS.")
            return
            
        start_pose = msg.poses[0].pose.position
        goal_pose = msg.poses[1].pose.position
        
        gx_s, gy_s = self.world_to_grid(start_pose.x, start_pose.y)
        gx_g, gy_g = self.world_to_grid(goal_pose.x, goal_pose.y)
        
        if not self.is_valid_point(gx_s, gy_s) or not self.is_valid_point(gx_g, gy_g):
            self.get_logger().error("Start or Goal is invalid (Collision/Out of bounds)!")
            return

        self.get_logger().info("Replanning Triggered...")
        
        # Chạy RRT* (Lưu ý truyền đúng tham số output_file)
        rrt = RRTStarAlgorithm([gx_s, gy_s], [gx_g, gy_g], 1000, 4, 20, 10, self.grid, 
                               resolution=self.resolution, origin=self.origin, output_file=self.csv_path)
        
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
                    raw_path = rrt.find_path_2(new_node)
                    
                    # Làm mượt + Lưu CSV (Hàm này trong rrt.py đã tự lưu file)
                    final_path = rrt.post_processing(raw_path)
                    
                    # Vẽ đường lên RViz
                    coords = [(n.x, n.y) for n in final_path] + [(gx_g, gy_g)]
                    self.viz.set_path(coords)
                    
                    # Gửi tín hiệu thành công cho Controller
                    self.success_pub.publish(Bool(data=True))
                    self.get_logger().info(f"Path Found & Saved to {self.csv_path}")
                    found = True
                    break
        
        if not found:
            self.get_logger().warn("RRT Failed to find path.")

    def world_to_grid(self, x, y):
        return int((x - self.origin[0]) / self.resolution), int((y - self.origin[1]) / self.resolution)

    def is_valid_point(self, gx, gy):
        if not (0 <= gx < self.width and 0 <= gy < self.height): return False
        return self.grid[gx, gy] != 100 and self.grid[gx, gy] != -1

def main(args=None):
    rclpy.init(args=args)
    node = RRTPlannerNode()
    rclpy.spin(node)
    node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()