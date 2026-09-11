#!/usr/bin/env python3
"""
visualize_existing_waypoints.py
───────────────────────────────
ROS 2 Node đọc file CSV waypoint có sẵn và publish Marker hiển thị đường đua trên RViz.
Dùng độc lập không ghi đè file CSV.
"""

import os
import csv
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from ament_index_python.packages import get_package_share_directory


class VisualizeExistingWaypoints(Node):
    def __init__(self):
        super().__init__('visualize_existing_waypoints_node')

        # Đọc parameter đường dẫn file CSV
        pkg_share = get_package_share_directory('waypoint')
        default_dir = os.path.join(pkg_share, 'f1tenth_waypoint_generator', 'racelines')
        default_csv = os.path.join(default_dir, 'f1tenth_waypoint.csv')

        self.declare_parameter('path_to_csvFile', default_dir)
        self.declare_parameter('csvFile_name', 'f1tenth_waypoint.csv')
        self.declare_parameter('marker_topic', '/f1tenth_waypoint_marker')
        self.declare_parameter('frame_id', 'map')

        csv_path = self.get_parameter('path_to_csvFile').value
        csv_name = self.get_parameter('csvFile_name').value
        self.marker_topic = self.get_parameter('marker_topic').value
        self.frame_id = self.get_parameter('frame_id').value

        self.full_csv_path = os.path.join(csv_path, csv_name)

        # Fallback paths nếu không thấy ở đường dẫn chính
        if not os.path.exists(self.full_csv_path):
            home = os.path.expanduser('~')
            fallbacks = [
                default_csv,
                os.path.join(home, 'f1_ws/src/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv'),
                os.path.join(home, 'f1_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv'),
                os.path.join(home, 'f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/f1tenth_waypoint.csv'),
                os.path.join(home, 'Desktop/f1tenth_waypoint.csv'),
            ]
            for fb in fallbacks:
                if os.path.exists(fb):
                    self.full_csv_path = fb
                    break

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.marker_pub = self.create_publisher(Marker, self.marker_topic, latched_qos)

        self.waypoints = []
        self.load_waypoints()

        # Timer 1Hz publish
        self.timer = self.create_timer(1.0, self.publish_marker)

        self.get_logger().info("=========================================")
        self.get_logger().info(" VISUALIZE EXISTING WAYPOINTS NODE STARTED")
        self.get_logger().info(f" CSV Path:     {self.full_csv_path}")
        self.get_logger().info(f" Marker Topic: {self.marker_topic}")
        self.get_logger().info(f" Waypoints:    {len(self.waypoints)}")
        self.get_logger().info("=========================================")

    def load_waypoints(self):
        if not os.path.exists(self.full_csv_path):
            self.get_logger().error(f"Cannot find CSV waypoint file: {self.full_csv_path}")
            return

        points = []
        try:
            with open(self.full_csv_path, 'r') as f:
                reader = csv.reader(f)
                first_row = next(reader, None)
                if first_row:
                    try:
                        points.append([float(first_row[0]), float(first_row[1])])
                    except ValueError:
                        pass
                for row in reader:
                    if len(row) >= 2:
                        try:
                            points.append([float(row[0]), float(row[1])])
                        except ValueError:
                            continue
            self.waypoints = points
            self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints.")
        except Exception as e:
            self.get_logger().error(f"Error loading CSV: {e}")

    def publish_marker(self):
        if len(self.waypoints) == 0:
            self.load_waypoints()
            if len(self.waypoints) == 0:
                return

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "f1tenth_waypoint"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.08
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        # Màu xanh lá green y chang waypoint_generator_node gốc
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

        if len(self.waypoints) > 2 and math.dist(self.waypoints[0], self.waypoints[-1]) < 3.0:
            p_first = Point()
            p_first.x = float(self.waypoints[0][0])
            p_first.y = float(self.waypoints[0][1])
            p_first.z = 0.0
            marker.points.append(p_first)

        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = VisualizeExistingWaypoints()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
