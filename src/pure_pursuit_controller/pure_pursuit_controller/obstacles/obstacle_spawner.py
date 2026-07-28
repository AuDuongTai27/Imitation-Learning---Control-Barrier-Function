#!/usr/bin/env python3
"""
obstacle_spawner.py
───────────────────
ROS 2 Node lắng nghe sự kiện click chuột trong RViz (/clicked_point)
và sinh ra một vật cản ảo (Marker) tại vị trí click.

Đồng thời, node này đóng vai trò là bộ lọc LiDAR:
1. Subcribe `/scan_raw` từ simulator.
2. Dùng TF để chuyển đổi tọa độ vật cản từ `map` sang hệ tọa độ của cảm biến LiDAR (laser frame).
3. Sử dụng thuật toán giao cắt Tia (Ray) và Hình tròn (Cylinder) để cập nhật dữ liệu ranges của scan.
4. Publish scan đã chèn vật cản lên topic `/scan` để các thuật toán tự lái thấy được vật cản như vật thể thật vật lý!
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
import math
import numpy as np

# TF2 imports để biến đổi tọa độ
from tf2_ros import Buffer, TransformListener, TransformException

class ObstacleSpawnerNode(Node):
    def __init__(self):
        super().__init__('obstacle_spawner_node')
        
        # --- Config ---
        self.obstacle_radius = 0.2  # Bán kính vật cản (30cm, đường kính 60cm)
        
        # --- TF2 Init ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # --- Pub / Sub ---
        # Lắng nghe click từ RViz (Nút 'Publish Point' trên thanh công cụ RViz)
        self.click_sub = self.create_subscription(PointStamped, '/clicked_point', self.click_callback, 10)
        
        # Lắng nghe scan thô từ simulator (đã cấu hình qua sim.yaml là scan_raw)
        self.scan_sub = self.create_subscription(LaserScan, '/scan_raw', self.scan_callback, 10)
        
        # Publish Marker trực quan hóa lên RViz
        self.marker_pub = self.create_publisher(Marker, '/sim_obstacle', 10)
        
        # Publish LaserScan đã chèn vật cản cho xe
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        
        # Trạng thái vật cản hiện tại (tọa độ trong map frame)
        self.obstacle_active = False
        self.obstacle_x = 0.0
        self.obstacle_y = 0.0
        
        self.get_logger().info("=========================================")
        self.get_logger().info(" OBSTACLE SPAWER & LIDAR INJECTOR READY")
        self.get_logger().info(" Instructions: Click 'Publish Point' in RViz toolbar")
        self.get_logger().info(" and click on the map to place/remove an obstacle.")
        self.get_logger().info("=========================================")

    def click_callback(self, msg: PointStamped):
        x = msg.point.x
        y = msg.point.y
        
        # Tính khoảng cách tới vật cản hiện tại (nếu có)
        dist_to_current = math.hypot(x - self.obstacle_x, y - self.obstacle_y)
        
        # Logic Toggle:
        # Nếu click vào điểm rất gần vật cản hiện tại (< 0.6m), ta sẽ XÓA vật cản đó đi.
        if self.obstacle_active and dist_to_current < 0.6:
            self.delete_obstacle(msg.header)
        else:
            self.spawn_obstacle(x, y, msg.header)

    def spawn_obstacle(self, x, y, header):
        self.obstacle_active = True
        self.obstacle_x = x
        self.obstacle_y = y
        
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "sim_obstacles"
        marker.id = 999
        marker.type = Marker.CYLINDER  # Hình trụ
        marker.action = Marker.ADD     # Thêm mới / Di chuyển
        
        # Vị trí đặt vật cản
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.5  # Đặt trọng tâm cao 0.5m để tiếp đất hoàn hảo
        marker.pose.orientation.w = 1.0
        
        # Kích thước vật cản
        marker.scale.x = self.obstacle_radius * 2.0  # Đường kính 60cm
        marker.scale.y = self.obstacle_radius * 2.0  # Đường kính 60cm
        marker.scale.z = 1.0                         # Cao 1m
        
        # Màu sắc (Màu Đỏ nổi bật)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        self.marker_pub.publish(marker)
        self.get_logger().info(f"🟢 Spawning Obstacle at: x={x:.2f}, y={y:.2f}")

    def delete_obstacle(self, header):
        self.obstacle_active = False
        
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "sim_obstacles"
        marker.id = 999
        marker.action = Marker.DELETE  # Xóa vật cản
        
        self.marker_pub.publish(marker)
        self.get_logger().info("🔴 Removed Obstacle.")

    def scan_callback(self, msg: LaserScan):
        """ Nhận scan thô, chèn vật cản ảo nếu hoạt động, rồi publish sang /scan """
        if not self.obstacle_active:
            # Nếu không có vật cản, chỉ cần chuyển tiếp (passthrough) scan thô sang /scan
            self.scan_pub.publish(msg)
            return

        # Tạo bản sao của scan thô để chỉnh sửa
        modified_scan = msg
        ranges = np.array(msg.ranges)
        
        # Lấy góc quét của từng tia
        angles = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min
        
        try:
            # Tra cứu vị trí của LiDAR (frame_id của scan) so với bản đồ (map)
            transform = self.tf_buffer.lookup_transform(
                msg.header.frame_id, 
                "map", 
                rclpy.time.Time()
            )
            
            # Tọa độ tịnh tiến
            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            
            # Góc quay quaternion -> yaw
            q = transform.transform.rotation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            # 1. Chuyển đổi tọa độ vật cản từ 'map' sang local frame của 'laser'
            # P_local = R_map_to_laser * P_map + T_map_to_laser
            ox_local = math.cos(yaw) * self.obstacle_x - math.sin(yaw) * self.obstacle_y + tx
            oy_local = math.sin(yaw) * self.obstacle_x + math.cos(yaw) * self.obstacle_y + ty
            
            # 2. Thuật toán giao cắt tia LiDAR và hình tròn (Cylinder)
            # Phương trình tia: P(t) = t * [cos(phi), sin(phi)]
            # Khoảng cách từ tâm hình tròn tới gốc laser: (ox_local, oy_local)
            # Hệ số bậc hai: t^2 + b*t + c = 0
            # Với: b = -2 * (ox_local*cos(phi) + oy_local*sin(phi))
            #      c = ox_local^2 + oy_local^2 - R^2
            cos_angles = np.cos(angles)
            sin_angles = np.sin(angles)
            
            b = -2.0 * (ox_local * cos_angles + oy_local * sin_angles)
            c = ox_local**2 + oy_local**2 - self.obstacle_radius**2
            
            discriminant = b**2 - 4.0 * c
            
            # Tạo mask lọc các tia cắt qua hình tròn (discriminant >= 0)
            valid_intersection_mask = discriminant >= 0
            
            if np.any(valid_intersection_mask):
                # Tính 2 nghiệm giao điểm t1, t2
                sqrt_disc = np.sqrt(discriminant[valid_intersection_mask])
                t1 = (-b[valid_intersection_mask] - sqrt_disc) / 2.0
                t2 = (-b[valid_intersection_mask] + sqrt_disc) / 2.0
                
                # Tìm khoảng cách giao điểm gần nhất ở phía trước tia (t > 0)
                t_intersection = np.minimum(t1, t2)
                
                # Cập nhật khoảng cách ranges nếu khoảng cách giao cắt nhỏ hơn giá trị scan hiện tại
                orig_ranges = ranges[valid_intersection_mask]
                updated_ranges = np.where(
                    (t_intersection > 0.05) & (t_intersection < orig_ranges), 
                    t_intersection, 
                    orig_ranges
                )
                ranges[valid_intersection_mask] = updated_ranges
            
            modified_scan.ranges = ranges.tolist()
            
        except TransformException as e:
            # Nếu chưa tra cứu được TF, chuyển tiếp scan thô để tránh làm đứng hệ thống
            pass
        
        self.scan_pub.publish(modified_scan)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleSpawnerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
