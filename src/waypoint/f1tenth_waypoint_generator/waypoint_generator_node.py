#!/usr/bin/env python3
import sys
from ament_index_python import get_package_prefix 

# Đảm bảo Python có thể tìm thấy các module trong package của bạn
# Lưu ý: Sửa lỗi gõ chữ từ "generation" thành "generator" để khớp với tên thư mục

# site_packages_path = f'{get_package_prefix("f1tenth_waypoint_generator")}/lib/python3.8/site-packages'
# if site_packages_path not in sys.path: 
#     sys.path.append(site_packages_path)


import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped

# Chỉ import những gì cần thiết cho clicked_waypoint
from f1tenth_waypoint_generator.record_clicked_waypoints import ClickedWaypoint

class WaypointGenerator(Node):
    def __init__(self):
        """
        Hàm khởi tạo được đơn giản hóa, chỉ tập trung vào clicked_waypoint.
        """
        super().__init__('waypoint_generator_node')
        self.get_logger().info("Khởi tạo node Waypoint Generator chỉ dành cho Clicked Points.")

        # Khởi tạo đối tượng xử lý waypoint ngay bên trong node
        # Ta truyền 'self' vì bản thân class này là một Node, ClickedWaypoint có thể dùng nó để logging,...
        self.clicked_waypoint_handler = ClickedWaypoint(self)

        # Khai báo và lấy tham số chvisualization_marker_arrayo topic của clicked_point
        self.declare_parameter('clicked_point_topic', '/clicked_point')
        topic_name = self.get_parameter('clicked_point_topic').get_parameter_value().string_value
        self.get_logger().info(f"Đang lắng nghe trên topic: '{topic_name}'")

        # Tạo subscriber để lắng nghe các điểm được click trong RViz
        self.clicked_point_sub = self.create_subscription(
            PointStamped,
            topic_name,
            self.waypoint_callback,
            10
        )
        
        # Tạo publisher để hiển thị các điểm waypoint dưới dạng Marker trong RViz
        self.waypoint_marker_pub = self.create_publisher(Marker, '/f1tenth_waypoint_marker', 10)
    
    def waypoint_callback(self, msg: PointStamped) -> None:
        """
        Callback này chỉ xử lý tin nhắn PointStamped từ topic clicked_point.
        """
        self.get_logger().info(f"Nhận được điểm được click: (x={msg.point.x:.2f}, y={msg.point.y:.2f})")
        
        # Gọi phương thức xử lý từ class ClickedWaypoint để lưu và hiển thị marker
        self.clicked_waypoint_handler.clicked_point_callback(msg, self.waypoint_marker_pub)


def main(args=None):
    rclpy.init(args=args)

    # Khởi tạo và chạy node đã được đơn giản hóa
    waypoint_generator_node = WaypointGenerator()
    
    try:
        rclpy.spin(waypoint_generator_node)
    except KeyboardInterrupt:
        print("Đã nhận tín hiệu ngắt, đang tắt node...")
    
    # Dọn dẹp
    waypoint_generator_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()