#!/usr/bin/env python3
"""
waypoint_visualizer.py
──────────────────────
ROS 2 Node visualization các điểm waypoint từ file CSV lên RViz.
Không đè/ghi mới file CSV như waypoint_generator_node.
Hiển thị liên tục trên topic `/f1tenth_waypoint_marker` (chuẩn RViz).
"""

import os
import csv
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


def resolve_waypoint_path(given_path=""):
    if given_path and os.path.exists(given_path):
        return given_path
    
    home = os.path.expanduser('~')
    candidates = [
        # Sim / Docker Paths
        "/sim_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv",
        "/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/f1tenth_waypoint.csv",
        "/sim_ws/src/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv",
        # Host Paths
        os.path.join(home, "f1_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"),
        os.path.join(home, "f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/f1tenth_waypoint.csv"),
        os.path.join(home, "f1_ws/src/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"),
        os.path.join(home, "Desktop/f1tenth_waypoint.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return given_path


class WaypointVisualizerNode(Node):
    def __init__(self):
        super().__init__('waypoint_visualizer_node')

        # --- Declare ROS 2 Parameters ---
        self.declare_parameter('waypoint_path', '/sim_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv')
        self.declare_parameter('marker_topic', '/f1tenth_waypoint_marker')
        self.declare_parameter('marker_array_topic', '/f1tenth_waypoint_array')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('publish_rate', 1.0)  # Hz
        self.declare_parameter('line_width', 0.08)
        self.declare_parameter('point_size', 0.12)

        # Read Parameters
        raw_path = self.get_parameter('waypoint_path').value
        self.marker_topic = self.get_parameter('marker_topic').value
        self.marker_array_topic = self.get_parameter('marker_array_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.line_width = self.get_parameter('line_width').value
        self.point_size = self.get_parameter('point_size').value

        self.waypoint_path = resolve_waypoint_path(raw_path)

        # Transient Local QoS cho Marker pub để RViz nhận ngay khi mở sau
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # Publishers
        self.marker_pub = self.create_publisher(Marker, self.marker_topic, latched_qos)
        self.marker_array_pub = self.create_publisher(MarkerArray, self.marker_array_topic, latched_qos)

        self.waypoints = []
        self.last_mtime = 0.0

        # Load waypoints
        self.reload_waypoints()

        # Timer publish định kỳ
        timer_period = 1.0 / max(self.publish_rate, 0.1)
        self.timer = self.create_timer(timer_period, self.publish_callback)

        self.get_logger().info("=========================================")
        self.get_logger().info(" WAYPOINT VISUALIZER NODE STARTED")
        self.get_logger().info(f" Waypoint File Path: {self.waypoint_path}")
        self.get_logger().info(f" Marker Topic:       {self.marker_topic}")
        self.get_logger().info(f" Frame ID:           {self.frame_id}")
        self.get_logger().info(f" Loaded Points:      {len(self.waypoints)}")
        self.get_logger().info("=========================================")

    def reload_waypoints(self):
        if not self.waypoint_path or not os.path.exists(self.waypoint_path):
            self.get_logger().warn(f"Waypoint file not found: '{self.waypoint_path}'")
            return False

        try:
            mtime = os.path.getmtime(self.waypoint_path)
            if mtime == self.last_mtime and len(self.waypoints) > 0:
                return True

            points = []
            with open(self.waypoint_path, 'r') as f:
                reader = csv.reader(f)
                first_row = next(reader, None)
                if first_row:
                    try:
                        points.append([float(first_row[0]), float(first_row[1])])
                    except ValueError:
                        pass  # Skip header row if string
                for row in reader:
                    if len(row) >= 2:
                        try:
                            points.append([float(row[0]), float(row[1])])
                        except ValueError:
                            continue

            if len(points) > 0:
                self.waypoints = points
                self.last_mtime = mtime
                self.get_logger().info(f"Successfully loaded {len(self.waypoints)} waypoints from CSV.")
                return True
            else:
                self.get_logger().warn(f"CSV file '{self.waypoint_path}' is empty or invalid.")
                return False

        except Exception as e:
            self.get_logger().error(f"Error loading waypoints: {e}")
            return False

    def create_line_strip_marker(self):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "f1tenth_raceline"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = float(self.line_width)
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        # Màu xanh lá mượt (Green) giống hệt waypoint_generator
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        for pt in self.waypoints:
            p = Point()
            p.x = float(pt[0])
            p.y = float(pt[1])
            p.z = 0.0
            marker.points.append(p)

        # Nối khép kín nếu điểm đầu và cuối gần nhau
        if len(self.waypoints) > 2:
            d_loop = math.dist(self.waypoints[0], self.waypoints[-1])
            if d_loop < 3.0:
                p_first = Point()
                p_first.x = float(self.waypoints[0][0])
                p_first.y = float(self.waypoints[0][1])
                p_first.z = 0.0
                marker.points.append(p_first)

        return marker

    def create_points_marker(self):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "f1tenth_waypoint_nodes"
        marker.id = 1
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD

        marker.scale.x = float(self.point_size)
        marker.scale.y = float(self.point_size)
        marker.scale.z = float(self.point_size)

        # Màu vàng nổi bật cho điểm nút waypoint
        marker.color.a = 0.9
        marker.color.r = 1.0
        marker.color.g = 0.8
        marker.color.b = 0.0

        for pt in self.waypoints:
            p = Point()
            p.x = float(pt[0])
            p.y = float(pt[1])
            p.z = 0.02
            marker.points.append(p)

        return marker

    def publish_callback(self):
        # Thử reload nếu file CSV cập nhật
        self.reload_waypoints()

        if len(self.waypoints) == 0:
            return

        # Publish single Line Strip Marker (chuẩn cũ)
        line_marker = self.create_line_strip_marker()
        self.marker_pub.publish(line_marker)

        # Publish MarkerArray (LINE_STRIP + SPHERE_LIST nút điểm)
        points_marker = self.create_points_marker()
        marker_array = MarkerArray()
        marker_array.markers.append(line_marker)
        marker_array.markers.append(points_marker)
        self.marker_array_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Waypoint Visualizer Node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
