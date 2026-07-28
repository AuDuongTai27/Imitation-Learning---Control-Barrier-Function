#!/usr/bin/env python3
"""
dagger_inference_sim.py
───────────────────────
ROS 2 Node chạy Suy luận mô hình Imitation Learning tích hợp cơ chế DAgger (Cứu nét) trong môi trường mô phỏng (f1tenth_gym_ros).

Logic:
  1. Đọc và giải mã dữ liệu LiDAR từ `/scan`, tiền xử lý về 60 beams.
  2. Lấy tọa độ X,Y của xe từ `/ego_racecar/odom` để tính toán khoảng cách lệch tâm đường (CTE).
  3. Tính toán trước lệnh lái tối ưu của Chuyên gia (Pure Pursuit - Ackermann) theo quỹ đạo chuẩn (waypoints).
  4. Nếu CTE < 0.15m:
     - Dùng AI Model (PyTorch) điều khiển xe chạy tự động (publish tới `/drive`).
  5. Nếu CTE >= 0.15m:
     - Kích hoạt chế độ cứu nét: Đè lệnh lái của Pure Pursuit lên `/drive` để cứu xe.
     - Thu thập đồng thời dữ liệu [LiDAR hiện tại + Lệnh lái của Chuyên gia] lưu lại vào bộ đệm.
     - Khi bộ đệm gom đủ 50 mẫu, lưu tiếp vào `dagger_dataset_sim.csv`.
"""

import os
os.environ["ORT_DISABLE_CPU_AFFINITY"] = "1"
import csv
import math
import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

# Import ONNX Runtime cho phần suy luận mô hình
try:
    import onnxruntime as ort
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False


def resolve_model_path(path):
    if os.path.exists(path):
        return path
    dirname, filename = os.path.split(path)
    alt_path = os.path.join(dirname, 'models', filename)
    if os.path.exists(alt_path):
        return alt_path
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    alt_path2 = os.path.join(curr_dir, '..', 'models', filename)
    if os.path.exists(alt_path2):
        return os.path.abspath(alt_path2)
    return path


def resolve_dataset_path(path):
    if os.path.exists(path):
        return path
    dirname, filename = os.path.split(path)
    alt_path = os.path.join(dirname, 'datasets', filename)
    if os.path.exists(alt_path):
        return alt_path
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    alt_path2 = os.path.join(curr_dir, '..', 'datasets', filename)
    if os.path.exists(alt_path2):
        return os.path.abspath(alt_path2)
    return path


class DaggerInferenceSimNode(Node):
    def __init__(self):
        super().__init__('dagger_inference_sim_node')

        # --- 1. Parameters ---
        default_model = '/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/models/dagger_model_sim.onnx'
        default_dataset = '/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/datasets/dagger_dataset_sim.csv'

        self.declare_parameter('model_path', resolve_model_path(default_model))
        self.declare_parameter('waypoint_path', '/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/f1tenth_waypoint.csv')
        self.declare_parameter('dataset_path', resolve_dataset_path(default_dataset))
        
        self.declare_parameter('target_beams', 60)
        self.declare_parameter('lookahead_dist', 1.0)
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('expert_speed', 1.5)       # Vận tốc của chuyên gia (m/s)
        self.declare_parameter('ai_speed', 1.5)           # Vận tốc tối đa của AI (m/s)
        self.declare_parameter('cte_threshold', 0.15)     # Ngưỡng cứu nét (m)
        self.declare_parameter('buffer_size', 50)
        self.declare_parameter('max_range', 10.0)

        self.model_path = resolve_model_path(self.get_parameter('model_path').value)
        self.waypoint_path = self.get_parameter('waypoint_path').value
        self.dataset_path = resolve_dataset_path(self.get_parameter('dataset_path').value)
        
        self.target_beams = self.get_parameter('target_beams').value
        self.lookahead_dist = self.get_parameter('lookahead_dist').value
        self.wheelbase = self.get_parameter('wheelbase').value
        self.expert_speed = self.get_parameter('expert_speed').value
        self.ai_speed = self.get_parameter('ai_speed').value
        self.cte_threshold = self.get_parameter('cte_threshold').value
        self.buffer_size = self.get_parameter('buffer_size').value
        self.max_range = self.get_parameter('max_range').value

        # --- 2. Load ONNX Model ---
        self.ort_session = None
        if _HAS_ONNX:
            if os.path.exists(self.model_path) and self.model_path.endswith('.onnx'):
                try:
                    self.ort_session = ort.InferenceSession(self.model_path)
                    self.get_logger().info(f"Successfully loaded ONNX model from {self.model_path}")
                except Exception as e:
                    self.get_logger().error(f"Failed to load ONNX model: {e}")
            else:
                self.get_logger().warn(f"ONNX Model file not found at {self.model_path} or invalid format. Will ONLY run on Expert mode.")
        else:
            self.get_logger().error("ONNX Runtime is not installed in this environment. Running in EXPERT-ONLY mode. Install it with: pip install onnxruntime")

        # --- 3. Waypoint & Pure Pursuit States ---
        self.waypoints = self.load_waypoints(self.waypoint_path)
        self.last_idx = 0
        
        # --- 4. State & Buffer Variables ---
        self.car_x = 0.0
        self.car_y = 0.0
        self.car_yaw = 0.0
        self.odom_received = False
        
        self.buffer = []
        self.lock = threading.Lock()
        self.total_saved_samples = 0

        # --- 5. Pub/Sub ---
        odom_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, odom_qos)

        self.get_logger().info("=========================================")
        self.get_logger().info(" DAGGER INFERENCE SIM NODE STARTED")
        self.get_logger().info(f" CTE Threshold: {self.cte_threshold}m")
        self.get_logger().info("=========================================")

    def load_waypoints(self, file_path):
        """Tải các điểm waypoint từ file CSV, nếu không có sẽ tự tạo đường tròn ảo để chạy thử"""
        if os.path.exists(file_path):
            try:
                points = []
                with open(file_path, 'r') as f:
                    reader = csv.reader(f)
                    first_row = next(reader, None)
                    if first_row:
                        try:
                            points.append([float(first_row[0]), float(first_row[1])])
                        except ValueError:
                            pass # Bỏ qua header
                    for row in reader:
                        if len(row) >= 2:
                            points.append([float(row[0]), float(row[1])])
                self.get_logger().info(f"Loaded {len(points)} waypoints from CSV.")
                return np.array(points)
            except Exception as e:
                self.get_logger().error(f"Error loading waypoints: {e}")

        # Fallback: Tạo quỹ đạo hình tròn bán kính 4m tâm tại (0, 3) để test
        self.get_logger().warn("Waypoint file not found! Generating circular fallback waypoints.")
        theta = np.linspace(0, 2*np.pi, 200)
        r = 4.0
        points = np.stack([r * np.cos(theta), r * np.sin(theta) + 3.0], axis=1)
        return points

    def odom_callback(self, msg: Odometry):
        """Cập nhật vị trí hiện tại của xe và góc quay (yaw)"""
        with self.lock:
            self.car_x = msg.pose.pose.position.x
            self.car_y = msg.pose.pose.position.y
            
            # Quaternion -> Yaw
            q = msg.pose.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.car_yaw = math.atan2(siny_cosp, cosy_cosp)
            self.odom_received = True

    def scan_callback(self, msg: LaserScan):
        """Xử lý LiDAR, tính sai lệch CTE, quyết định điều khiển qua AI hoặc Expert"""
        with self.lock:
            if not self.odom_received:
                return
            curr_x = self.car_x
            curr_y = self.car_y
            curr_yaw = self.car_yaw

        # 1. Tính toán sai lệch khoảng cách (Cross-track Error - CTE) đến quỹ đạo
        cte, nearest_idx = self.calculate_cross_track_error(curr_x, curr_y)

        # 2. Tính toán lệnh lái dự phòng từ Chuyên gia (Pure Pursuit - Ackermann)
        expert_speed, expert_steer = self.calculate_pure_pursuit(curr_x, curr_y, curr_yaw, nearest_idx)

        # 3. Tiền xử lý dữ liệu scan
        preprocessed_scan = self.preprocess_scan(msg)

        # 4. Trạng thái cứu nét hoặc AI
        if self.ort_session is None or cte >= self.cte_threshold:
            # --- CHẾ ĐỘ CHUYÊN GIA CỨU NÉT (OVERRIDE) ---
            self.publish_drive(expert_speed, expert_steer)
            self.get_logger().warn(f"[SIM - OVERRIDE] CTE: {cte:.3f}m >= {self.cte_threshold}m. Expert driving.", throttle_duration_sec=1.0)
            
            # Lưu lại dữ liệu lúc cứu nét (DAgger data aggregation)
            with self.lock:
                row = list(preprocessed_scan) + [expert_speed, expert_steer]
                self.buffer.append(row)
                
                if len(self.buffer) >= self.buffer_size:
                    buffer_to_save = list(self.buffer)
                    self.buffer.clear()
                    threading.Thread(target=self._flush_buffer, args=(buffer_to_save,), daemon=True).start()
        else:
            # --- CHẾ ĐỘ AI SUY LUẬN TỰ ĐỘNG ---
            ai_speed, ai_steer = self.run_model_inference(preprocessed_scan)
            self.publish_drive(ai_speed, ai_steer)
            self.get_logger().info(f"[SIM - AI DRIVING] CTE: {cte:.3f}m < {self.cte_threshold}m.", throttle_duration_sec=2.0)

    def calculate_cross_track_error(self, car_x, car_y):
        """Tính khoảng cách vuông góc ngắn nhất từ xe tới đường đi"""
        num_pts = len(self.waypoints)
        search_len = min(50, num_pts)
        
        indices = [(self.last_idx + i) % num_pts for i in range(search_len)]
        search_points = self.waypoints[indices]
        
        dists = np.linalg.norm(search_points - np.array([car_x, car_y]), axis=1)
        min_local_idx = np.argmin(dists)
        nearest_idx = indices[min_local_idx]
        self.last_idx = nearest_idx

        # Lấy 2 điểm tạo thành đoạn thẳng quỹ đạo chuẩn
        A = self.waypoints[nearest_idx]
        B = self.waypoints[(nearest_idx + 1) % num_pts]
        P = np.array([car_x, car_y])

        AB = B - A
        AP = P - A
        ab_len_sq = np.sum(AB**2)

        if ab_len_sq < 1e-6:
            cte = np.linalg.norm(AP)
        else:
            t = np.clip(np.dot(AP, AB) / ab_len_sq, 0.0, 1.0)
            closest_pt = A + t * AB
            cte = np.linalg.norm(P - closest_pt)

        return cte, nearest_idx

    def calculate_pure_pursuit(self, car_x, car_y, car_yaw, nearest_idx):
        """Thuật toán Pure Pursuit để tính lệnh lái Ackermann"""
        num_pts = len(self.waypoints)
        lookahead_idx = nearest_idx
        
        # Tìm điểm đích cách xe một khoảng lookahead_dist
        while True:
            lookahead_idx = (lookahead_idx + 1) % num_pts
            dist = math.hypot(car_x - self.waypoints[lookahead_idx][0], car_y - self.waypoints[lookahead_idx][1])
            if dist >= self.lookahead_dist:
                target_pt = self.waypoints[lookahead_idx]
                break
            if lookahead_idx == nearest_idx:
                target_pt = self.waypoints[lookahead_idx]
                break

        # Chuyển điểm đích sang hệ tọa độ của xe (Car Frame)
        dx = target_pt[0] - car_x
        dy = target_pt[1] - car_y
        local_y = -dx * math.sin(car_yaw) + dy * math.cos(car_yaw)

        # Tính góc lái Pure Pursuit cho xe Ackermann: delta = atan(2 * L * y / ld^2)
        steering_angle = math.atan((2.0 * self.wheelbase * local_y) / (self.lookahead_dist**2))

        # Giới hạn góc lái tối đa (khoảng 24 độ)
        steering_angle = np.clip(steering_angle, -0.41, 0.41)

        # Giảm tốc khi cua gấp
        speed = self.expert_speed
        if abs(steering_angle) > 0.2:
            speed *= 0.7

        return speed, steering_angle

    def preprocess_scan(self, msg: LaserScan):
        """Crop góc quét về [-60, 60] độ và downsample về 60 beams"""
        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment
        crop_limit = math.radians(60.0)

        angles = np.arange(len(ranges)) * angle_increment + angle_min
        mask = (angles >= -crop_limit) & (angles <= crop_limit)

        if not np.any(mask):
            return np.ones(self.target_beams, dtype=np.float32) * self.max_range

        valid_ranges = ranges[mask]
        valid_angles = angles[mask]

        valid_ranges = np.where(np.isnan(valid_ranges) | np.isinf(valid_ranges), self.max_range, valid_ranges)
        valid_ranges = np.clip(valid_ranges, 0.0, self.max_range)

        target_angles = np.linspace(-crop_limit, crop_limit, self.target_beams)
        return np.interp(target_angles, valid_angles, valid_ranges)

    def run_model_inference(self, preprocessed_scan):
        """Dự đoán lệnh lái Ackermann qua ONNX Model"""
        if not _HAS_ONNX or self.ort_session is None:
            return 0.0, 0.0
            
        # Chuẩn hóa giống lúc train: chia 10.0
        norm_scan = (preprocessed_scan / 10.0).astype(np.float32)
        tensor_input = np.expand_dims(norm_scan, axis=0) # Shape (1, 60)
        
        # Chạy inference qua ONNX Runtime
        outputs = self.ort_session.run(None, {'input': tensor_input})
        output = outputs[0][0]
            
        # Output: [speed, steering_angle]
        speed = float(np.clip(output[0], 0.0, self.ai_speed))
        steering_angle = float(np.clip(output[1], -0.41, 0.41))
        return speed, steering_angle

    def publish_drive(self, speed, steering_angle):
        """Publish lệnh tới topic /drive (AckermannDriveStamped)"""
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'ego_racecar'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)

    def _flush_buffer(self, data_list):
        """Ghi dữ liệu Expert cứu nét vào dataset CSV"""
        if not os.path.exists(self.dataset_path):
            with open(self.dataset_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = [f'lidar_{i}' for i in range(self.target_beams)] + ['speed', 'steering_angle']
                writer.writerow(header)

        try:
            with open(self.dataset_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data_list)
            
            self.total_saved_samples += len(data_list)
            self.get_logger().info(f"[SIM - DAgger] Logged {len(data_list)} expert recovery samples. Total: {self.total_saved_samples}")
        except Exception as e:
            self.get_logger().error(f"Error saving DAgger dataset: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DaggerInferenceSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Shutting down DAgger simulation inference node.")
        node.publish_drive(0.0, 0.0)
        if len(node.buffer) > 0:
            node._flush_buffer(node.buffer)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
