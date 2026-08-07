#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math
import csv
import os
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        
        # --- 1. Parameters ---
        self.declare_parameter("waypoint_path", "/sim_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv")
        self.declare_parameter("lookahead_dist", 1.0)
        self.declare_parameter("max_speed", 3.0)
        
        self.csv_path = self.get_parameter("waypoint_path").value
        self.L = self.get_parameter("lookahead_dist").value
        self.max_speed = self.get_parameter("max_speed").value
        
        # --- 2. Load Data ---
        self.waypoints = self.load_waypoints(self.csv_path)
        self.last_idx = 0
        
        # --- 3. Pub/Sub ---
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization/markers', 10)
        self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)
        self.pub_text_marker = self.create_publisher(Marker, "/ten_xe", 10)
        self.pub_marker_3 = self.create_publisher(MarkerArray, "/publish_duong_di", 10)
        
        self.publish_static_path()
        self.get_logger().info(f"Pure Pursuit Ready. Points: {len(self.waypoints)}")

    def publish_duong_di(self):
        marker_array = MarkerArray()
        for i, point in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.1; marker.scale.y = 0.1; marker.scale.z = 0.1
            marker.color.a = 1.0; marker.color.r = 0.5; marker.color.g = 0.5
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker_array.markers.append(marker)
        self.pub_marker_3.publish(marker_array)

    def publish_ten_xe(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "text_info"
        marker.id = 999
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.scale.z = 1.0
        marker.color.a = 1.0; marker.color.r = 0.0; marker.color.g = 0.5; marker.color.b = 1.0
        marker.pose.position.x = x 
        marker.pose.position.y = y 
        marker.pose.position.z = 1.2
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.text = "EIU FABLAB"
        self.pub_text_marker.publish(marker)

    def load_waypoints(self, file_path):
        """Load waypoints from CSV"""
        points = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                next(reader, None) 
                for row in reader:
                    points.append([float(row[0]), float(row[1])])
        return np.array(points)

    def odom_callback(self, msg):
        if len(self.waypoints) == 0: return

        car_x = msg.pose.pose.position.x
        car_y = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        car_yaw = math.atan2(siny_cosp, cosy_cosp)

        target_point = self.get_target_point(car_x, car_y)
        steering_angle = self.calculate_steering(target_point, car_x, car_y, car_yaw)
        self.publish_drive(steering_angle)
        
        self.publish_dynamic_markers(target_point, car_x, car_y)
        self.publish_ten_xe(car_x, car_y)

    def get_target_point(self, car_x, car_y):
        """Find lookahead point"""
        num_pts = len(self.waypoints)
        search_len = 50
        
        indices = [(self.last_idx + i) % num_pts for i in range(search_len)]
        search_points = np.array([self.waypoints[i] for i in indices])
        
        dists = np.linalg.norm(search_points - np.array([car_x, car_y]), axis=1)
        min_local_idx = np.argmin(dists)
        
        nearest_idx = indices[min_local_idx]
        self.last_idx = nearest_idx
        
        lookahead_idx = nearest_idx
        while True:
            lookahead_idx = (lookahead_idx + 1) % num_pts
            dist = math.dist([car_x, car_y], self.waypoints[lookahead_idx])
            
            if dist >= self.L:
                return self.waypoints[lookahead_idx]
            
            if lookahead_idx == nearest_idx:
                return self.waypoints[lookahead_idx]

    def calculate_steering(self, target, car_x, car_y, car_yaw):
        """Calculate pure pursuit steering angle"""
        tx, ty = target
        dx = tx - car_x
        dy = ty - car_y
        
        target_y_local = dx * math.sin(-car_yaw) + dy * math.cos(-car_yaw)
        lookahead_dist = math.dist([car_x, car_y], target)
        wheelbase = 0.33
        
        return math.atan((2 * wheelbase * target_y_local) / (lookahead_dist**2))

    def publish_drive(self, angle):
        speed = self.max_speed
        angle = np.clip(angle, -0.35, 0.35)
            
        msg = AckermannDriveStamped()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(angle)
        self.drive_pub.publish(msg)

    def stop_vehicle(self):
        self.get_logger().warn("Stopping Vehicle...")
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        self.drive_pub.publish(msg)

    def create_marker(self, id, x, y, r, g, b, scale=0.3, type=Marker.SPHERE):
        m = Marker()
        m.header.frame_id = "map"
        m.id = id; m.type = type; m.action = Marker.ADD
        m.pose.position.x = float(x); m.pose.position.y = float(y)
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color.a = 1.0; m.color.r = r; m.color.g = g; m.color.b = b
        return m

    def publish_static_path(self):
        if len(self.waypoints) == 0: return
        marker_array = MarkerArray()
        path_marker = self.create_marker(999, 0, 0, 1.0, 0.0, 0.0, 0.1, Marker.SPHERE_LIST)
        for pt in self.waypoints:
            p = Point(); p.x = pt[0]; p.y = pt[1]
            path_marker.points.append(p)
        marker_array.markers.append(path_marker)
        self.marker_pub.publish(marker_array)

    def publish_dynamic_markers(self, target, car_x, car_y):
        marker_array = MarkerArray()
        marker_array.markers.append(self.create_marker(0, target[0], target[1], 0.0, 1.0, 0.0))
        marker_array.markers.append(self.create_marker(1, car_x, car_y, 0.0, 0.0, 1.0))
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_vehicle()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()