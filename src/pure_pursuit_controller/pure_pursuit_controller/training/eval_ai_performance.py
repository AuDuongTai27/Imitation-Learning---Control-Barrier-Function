#!/usr/bin/env python3
"""
eval_ai_performance.py
──────────────────────
ROS 2 Node đánh giá định lượng & trực quan hóa hiệu suất xe tự lái AI (Autonomous AI Evaluation Node).
Chạy node này SONG SONG khi xe đang suy luận AI (ai_inference_sim.py / ai_inference_sim_odom.py).

Các chỉ số nghiên cứu (Research Metrics) được tính toán tự động:
  1. Steering RMSE & MAE: Sai số bình phương trung bình & tuyệt đối giữa góc lái AI (AI command) vs Chuyên gia Pure Pursuit (Ground Truth).
  2. Cross-Track Error (CTE): Khoảng cách lệch làn thực tế của xe so với đường đua chuẩn (Raceline GT).
  3. Speed RMSE & MAE: Độ lệch vận tốc so với vận tốc chuyên gia đề xuất.
  4. 2D Spatial Trajectory Map: Quỹ đạo xe chạy thực tế (X-Y) so với đường chuẩn GT.

Khi dừng node (Ctrl+C), node sẽ xuất dữ liệu CSV (ai_eval_log.csv) và tự động tạo báo cáo nghiên cứu dạng HTML tương tác (ai_eval_report.html).
"""

import os
import csv
import math
import time
import json
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class AiPerformanceEvaluatorNode(Node):
    def __init__(self):
        super().__init__('ai_performance_evaluator_node')

        # --- 1. Parameters ---
        # Chỉ định đường dẫn ghi ra thư mục src/ (được mount với máy Host) thay vì install/
        if os.path.exists('/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller'):
            target_src_dir = '/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller'
        else:
            home_dir = os.path.expanduser('~')
            target_src_dir = os.path.join(home_dir, 'f1_ws/src/pure_pursuit_controller/pure_pursuit_controller')

        default_csv_path = os.path.join(target_src_dir, 'ai_eval_log.csv')
        default_html_path = os.path.join(target_src_dir, 'ai_eval_report.html')

        if os.path.exists('/sim_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv'):
            default_waypoint_path = '/sim_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv'
        else:
            home_dir = os.path.expanduser('~')
            default_waypoint_path = os.path.join(
                home_dir,
                'f1_ws/src/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv'
            )

        self.declare_parameter('waypoint_path', default_waypoint_path)
        self.declare_parameter('csv_output_path', default_csv_path)
        self.declare_parameter('html_output_path', default_html_path)
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('lookahead_dist', 1.0)
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('expert_speed', 3.0)

        self.waypoint_path = self.get_parameter('waypoint_path').value
        self.csv_output_path = self.get_parameter('csv_output_path').value
        self.html_output_path = self.get_parameter('html_output_path').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.drive_topic = self.get_parameter('drive_topic').value
        self.L = self.get_parameter('lookahead_dist').value
        self.wheelbase = self.get_parameter('wheelbase').value
        self.expert_speed = self.get_parameter('expert_speed').value

        # --- 2. Load Raceline GT Waypoints ---
        self.waypoints = self.load_waypoints(self.waypoint_path)
        self.last_waypoint_idx = 0

        # --- 3. Synchronized Log Buffer ---
        self.latest_ai_drive = None
        self.latest_ai_time = 0.0
        self.lock = threading.Lock()

        # Storage for evaluation: list of dicts
        # {"timestamp_s", "x", "y", "yaw", "cte", "steer_ai", "steer_expert", "steer_err", "speed_ai", "speed_expert"}
        self.eval_records = []
        self.start_time_ns = None

        # Prepare CSV file
        os.makedirs(os.path.dirname(self.csv_output_path), exist_ok=True)
        self.csv_file = open(self.csv_output_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp_s', 'x', 'y', 'yaw', 'cte_m',
            'steer_ai_rad', 'steer_expert_rad', 'steer_error_rad',
            'speed_ai_mps', 'speed_expert_mps'
        ])
        self.csv_file.flush()

        # --- 4. Subscriptions ---
        self.drive_sub = self.create_subscription(
            AckermannDriveStamped, self.drive_topic, self.drive_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, 10
        )

        self.get_logger().info("=========================================")
        self.get_logger().info(" 📈 AI RESEARCH PERFORMANCE EVALUATOR STARTED")
        self.get_logger().info(f" Waypoints Loaded: {len(self.waypoints)} points")
        self.get_logger().info(f" Odom Topic: {self.odom_topic} | Drive Topic: {self.drive_topic}")
        self.get_logger().info(f" Log CSV: {self.csv_output_path}")
        self.get_logger().info(f" HTML Report: {self.html_output_path}")
        self.get_logger().info(" Keep node running while AI is driving. Press Ctrl+C when done.")
        self.get_logger().info("=========================================")

    def load_waypoints(self, file_path):
        """Tải các điểm raceline chuẩn (Ground Truth) từ CSV"""
        points = []
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    reader = csv.reader(f)
                    first_row = next(reader, None)
                    if first_row:
                        try:
                            points.append([float(first_row[0]), float(first_row[1])])
                        except ValueError:
                            pass
                    for row in reader:
                        if len(row) >= 2:
                            points.append([float(row[0]), float(row[1])])
            except Exception as e:
                self.get_logger().error(f"Error loading waypoints: {e}")
        return np.array(points) if len(points) > 0 else np.empty((0, 2))

    def drive_callback(self, msg: AckermannDriveStamped):
        """Lưu lại góc lái và tốc độ do AI phát đi tại thời điểm mới nhất"""
        with self.lock:
            self.latest_ai_drive = msg
            self.latest_ai_time = time.monotonic()

    def odom_callback(self, msg: Odometry):
        """Nhận vị trí thực tế của xe, tính toán chỉ số chuyên gia GT và CTE"""
        now_ns = self.get_clock().now().nanoseconds
        if self.start_time_ns is None:
            self.start_time_ns = now_ns
        t_sec = round((now_ns - self.start_time_ns) / 1e9, 4)

        now_mono = time.monotonic()
        with self.lock:
            if self.latest_ai_drive is None or (now_mono - self.latest_ai_time) > 0.5:
                return
            ai_drive = self.latest_ai_drive
            steer_ai = float(ai_drive.drive.steering_angle)
            speed_ai = float(ai_drive.drive.speed)

        # Trích xuất trạng thái xe
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # 1. Tính Cross-Track Error (CTE) & Góc lái Pure Pursuit Chuyên gia (Ground Truth)
        cte, steer_expert, speed_expert = self.compute_expert_ground_truth(x, y, yaw)

        steer_error = steer_ai - steer_expert

        record = {
            "timestamp_s": t_sec,
            "x": round(x, 4),
            "y": round(y, 4),
            "yaw": round(yaw, 4),
            "cte_m": round(cte, 4),
            "steer_ai_rad": round(steer_ai, 4),
            "steer_expert_rad": round(steer_expert, 4),
            "steer_error_rad": round(steer_error, 4),
            "speed_ai_mps": round(speed_ai, 4),
            "speed_expert_mps": round(speed_expert, 4)
        }

        self.eval_records.append(record)

        # Ghi ngay vào CSV
        self.csv_writer.writerow([
            record["timestamp_s"], record["x"], record["y"], record["yaw"], record["cte_m"],
            record["steer_ai_rad"], record["steer_expert_rad"], record["steer_error_rad"],
            record["speed_ai_mps"], record["speed_expert_mps"]
        ])
        if len(self.eval_records) % 20 == 0:
            self.csv_file.flush()
            self.get_logger().info(
                f"[EVAL] Time: {t_sec:5.1f}s | CTE: {cte:.3f}m | Steer Err: {math.degrees(abs(steer_error)):.2f}°",
                throttle_duration_sec=1.5
            )

    def compute_expert_ground_truth(self, car_x, car_y, car_yaw):
        """Tính CTE và Góc lái Chuyên gia Pure Pursuit tại vị trí (car_x, car_y)"""
        if len(self.waypoints) == 0:
            return 0.0, 0.0, self.expert_speed

        num_pts = len(self.waypoints)
        search_len = min(50, num_pts)
        indices = [(self.last_waypoint_idx + i) % num_pts for i in range(search_len)]
        search_pts = self.waypoints[indices]

        dists = np.linalg.norm(search_pts - np.array([car_x, car_y]), axis=1)
        min_idx_in_search = np.argmin(dists)
        nearest_idx = indices[min_idx_in_search]
        self.last_waypoint_idx = nearest_idx

        # Cross-Track Error (CTE) = khoảng cách ngắn nhất từ xe tới raceline
        cte = float(dists[min_idx_in_search])

        # Tìm điểm Lookahead cách xe khoảng L
        lookahead_idx = nearest_idx
        while True:
            lookahead_idx = (lookahead_idx + 1) % num_pts
            dist_to_pt = math.dist([car_x, car_y], self.waypoints[lookahead_idx])
            if dist_to_pt >= self.L or lookahead_idx == nearest_idx:
                target_pt = self.waypoints[lookahead_idx]
                break

        # Tính góc lái Pure Pursuit của Chuyên gia
        dx = target_pt[0] - car_x
        dy = target_pt[1] - car_y
        map_steering_angle = math.atan2(dy, dx)
        alpha = map_steering_angle - car_yaw

        # Chuẩn hóa góc alpha về [-pi, pi]
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        steer_expert = math.atan2(2.0 * self.wheelbase * math.sin(alpha), self.L)
        steer_expert = np.clip(steer_expert, -0.41, 0.41)

        return cte, float(steer_expert), self.expert_speed

    def generate_html_report(self):
        """Sinh ra báo cáo nghiên cứu tương tác HTML hoành tráng (Chart.js + Statistical Metrics)"""
        if len(self.eval_records) == 0:
            self.get_logger().warn("No data recorded to generate evaluation report.")
            return

        records = self.eval_records
        steer_errs = np.array([r["steer_error_rad"] for r in records])
        ctes = np.array([r["cte_m"] for r in records])
        speeds_ai = np.array([r["speed_ai_mps"] for r in records])
        speeds_exp = np.array([r["speed_expert_mps"] for r in records])

        # Tính toán các chỉ số nghiên cứu (Research Metrics)
        steer_rmse_rad = float(np.sqrt(np.mean(steer_errs**2)))
        steer_rmse_deg = float(math.degrees(steer_rmse_rad))
        steer_mae_rad = float(np.mean(np.abs(steer_errs)))
        steer_mae_deg = float(math.degrees(steer_mae_rad))

        mean_cte = float(np.mean(ctes))
        max_cte = float(np.max(ctes))
        std_cte = float(np.std(ctes))

        speed_rmse = float(np.sqrt(np.mean((speeds_ai - speeds_exp)**2)))
        mean_speed_ai = float(np.mean(speeds_ai))
        total_time_s = records[-1]["timestamp_s"] - records[0]["timestamp_s"]

        # Lấy dữ liệu 2D Trajectory
        traj_x = [r["x"] for r in records]
        traj_y = [r["y"] for r in records]

        gt_x = self.waypoints[:, 0].tolist() if len(self.waypoints) > 0 else []
        gt_y = self.waypoints[:, 1].tolist() if len(self.waypoints) > 0 else []

        print("\n=======================================================")
        print(" 📊 AUTONOMOUS AI EVALUATION SUMMARY REPORT")
        print("=======================================================")
        print(f" ⏱ Total Evaluation Time: {total_time_s:.2f} s ({len(records)} samples)")
        print(f" 🎯 Steering Angle RMSE  : {steer_rmse_rad:.4f} rad ({steer_rmse_deg:.2f}°)")
        print(f" 🎯 Steering Angle MAE   : {steer_mae_rad:.4f} rad ({steer_mae_deg:.2f}°)")
        print(f" 🛣 Mean Cross-Track Error: {mean_cte:.4f} m (Max: {max_cte:.4f} m, Std: {std_cte:.4f} m)")
        print(f" ⚡ Average AI Speed      : {mean_speed_ai:.2f} m/s (Speed RMSE: {speed_rmse:.3f} m/s)")
        print("=======================================================\n")

        # Code HTML + Chart.js báo cáo nghiên cứu
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>F1TENTH Autonomous AI Research Evaluation Report</title>
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        h1 {{
            color: #818cf8;
            font-size: 32px;
            margin: 0 0 8px 0;
        }}
        .subtitle {{
            color: #94a3b8;
            font-size: 16px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #1e293b;
            border-radius: 12px;
            padding: 20px;
            border-left: 5px solid #6366f1;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }}
        .metric-title {{
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: 700;
            color: #38bdf8;
            margin-top: 8px;
        }}
        .metric-sub {{
            font-size: 13px;
            color: #a7f3d0;
            margin-top: 4px;
        }}
        .chart-card {{
            background: #1e293b;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid #334155;
        }}
        .chart-title {{
            font-size: 18px;
            font-weight: 600;
            color: #f1f5f9;
            margin-bottom: 16px;
        }}
        .chart-container {{
            position: relative;
            height: 380px;
            width: 100%;
        }}
        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }}
        @media (max-width: 900px) {{
            .two-col {{ grid-template-columns: 1fr; }}
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏎️ F1TENTH AUTONOMOUS AI EVALUATION REPORT</h1>
            <div class="subtitle">Quantitative Evaluation of End-to-End Neural Controller vs Expert Raceline</div>
        </div>

        <!-- Research Metrics Dashboard -->
        <div class="metrics-grid">
            <div class="metric-card" style="border-left-color: #6366f1;">
                <div class="metric-title">Steering RMSE</div>
                <div class="metric-value">{steer_rmse_deg:.2f}°</div>
                <div class="metric-sub">{steer_rmse_rad:.4f} rad</div>
            </div>
            <div class="metric-card" style="border-left-color: #8b5cf6;">
                <div class="metric-title">Steering MAE</div>
                <div class="metric-value">{steer_mae_deg:.2f}°</div>
                <div class="metric-sub">{steer_mae_rad:.4f} rad</div>
            </div>
            <div class="metric-card" style="border-left-color: #06b6d4;">
                <div class="metric-title">Mean Cross-Track Error</div>
                <div class="metric-value">{mean_cte:.3f} m</div>
                <div class="metric-sub">Max CTE: {max_cte:.3f} m (Std: {std_cte:.3f})</div>
            </div>
            <div class="metric-card" style="border-left-color: #10b981;">
                <div class="metric-title">Average Speed & Total Time</div>
                <div class="metric-value">{mean_speed_ai:.2f} m/s</div>
                <div class="metric-sub">Duration: {total_time_s:.1f}s ({len(records)} samples)</div>
            </div>
        </div>

        <!-- Chart 1: Steering Angle Comparison -->
        <div class="chart-card">
            <div class="chart-title">1. Steering Angle Trajectory: AI Model vs Expert Ground Truth</div>
            <div class="chart-container">
                <canvas id="steerChart"></canvas>
            </div>
        </div>

        <div class="two-col">
            <!-- Chart 2: Cross-Track Error (CTE) over time -->
            <div class="chart-card">
                <div class="chart-title">2. Cross-Track Error (CTE) Over Time (m)</div>
                <div class="chart-container">
                    <canvas id="cteChart"></canvas>
                </div>
            </div>

            <!-- Chart 3: 2D Spatial Map Trajectory -->
            <div class="chart-card">
                <div class="chart-title">3. 2D Spatial Trajectory Map (X-Y Plane)</div>
                <div class="chart-container">
                    <canvas id="spatialChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Chart 4: Steering Error Distribution -->
        <div class="chart-card">
            <div class="chart-title">4. Steering Error Distribution (AI Steering - Expert Steering)</div>
            <div class="chart-container">
                <canvas id="errorHistChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        const records = {json.dumps(records)};
        const gtX = {json.dumps(gt_x)};
        const gtY = {json.dumps(gt_y)};

        const timestamps = records.map(r => r.timestamp_s);
        const steerAI = records.map(r => r.steer_ai_rad);
        const steerExp = records.map(r => r.steer_expert_rad);
        const ctes = records.map(r => r.cte_m);
        const steerErrs = records.map(r => r.steer_error_rad);
        const carX = records.map(r => r.x);
        const carY = records.map(r => r.y);

        const commonOptions = (yLabel) => ({{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
                x: {{
                    title: {{ display: true, text: 'Time (s)', color: '#94a3b8' }},
                    grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                    ticks: {{ color: '#94a3b8' }}
                }},
                y: {{
                    title: {{ display: true, text: yLabel, color: '#94a3b8' }},
                    grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                    ticks: {{ color: '#94a3b8' }}
                }}
            }},
            plugins: {{ legend: {{ labels: {{ color: '#f8fafc' }} }} }}
        }});

        // 1. Steer Chart
        new Chart(document.getElementById('steerChart'), {{
            type: 'line',
            data: {{
                labels: timestamps,
                datasets: [
                    {{
                        label: 'AI Model Steering (rad)',
                        data: steerAI,
                        borderColor: '#38bdf8',
                        backgroundColor: 'rgba(56, 189, 248, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Expert Pure Pursuit (rad)',
                        data: steerExp,
                        borderColor: '#f43f5e',
                        borderWidth: 1.8,
                        borderDash: [4, 4],
                        pointRadius: 0
                    }}
                ]
            }},
            options: commonOptions('Steering Angle (rad)')
        }});

        // 2. CTE Chart
        new Chart(document.getElementById('cteChart'), {{
            type: 'line',
            data: {{
                labels: timestamps,
                datasets: [{{
                    label: 'Cross-Track Error (m)',
                    data: ctes,
                    borderColor: '#a855f7',
                    backgroundColor: 'rgba(168, 85, 247, 0.15)',
                    fill: true,
                    borderWidth: 2,
                    pointRadius: 0
                }}]
            }},
            options: commonOptions('CTE (m)')
        }});

        // 3. 2D Spatial Map Chart
        const carPathData = carX.map((x, i) => ({{ x: x, y: carY[i] }}));
        const gtPathData = gtX.map((x, i) => ({{ x: x, y: gtY[i] }}));

        new Chart(document.getElementById('spatialChart'), {{
            type: 'scatter',
            data: {{
                datasets: [
                    {{
                        label: 'Ground Truth Raceline',
                        data: gtPathData,
                        borderColor: 'rgba(148, 163, 184, 0.5)',
                        borderWidth: 1.5,
                        showLine: true,
                        pointRadius: 0
                    }},
                    {{
                        label: 'AI Car Trajectory',
                        data: carPathData,
                        borderColor: '#10b981',
                        borderWidth: 2.5,
                        showLine: true,
                        pointRadius: 0
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{ title: {{ display: true, text: 'X (m)', color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#94a3b8' }} }},
                    y: {{ title: {{ display: true, text: 'Y (m)', color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#94a3b8' }} }}
                }},
                plugins: {{ legend: {{ labels: {{ color: '#f8fafc' }} }} }}
            }}
        }});

        // 4. Steering Error Histogram Chart
        // Compute simple histogram bins
        const minErr = Math.min(...steerErrs);
        const maxErr = Math.max(...steerErrs);
        const numBins = 15;
        const binWidth = (maxErr - minErr) / numBins || 0.01;
        const binCounts = new Array(numBins).fill(0);
        const binLabels = [];

        for (let i = 0; i < numBins; i++) {{
            const bStart = minErr + i * binWidth;
            const bEnd = bStart + binWidth;
            binLabels.push(`${{bStart.toFixed(2)}} to ${{bEnd.toFixed(2)}}`);
        }}

        steerErrs.forEach(e => {{
            let idx = Math.floor((e - minErr) / binWidth);
            if (idx >= numBins) idx = numBins - 1;
            binCounts[idx]++;
        }});

        new Chart(document.getElementById('errorHistChart'), {{
            type: 'bar',
            data: {{
                labels: binLabels,
                datasets: [{{
                    label: 'Frequency Count',
                    data: binCounts,
                    backgroundColor: 'rgba(99, 102, 241, 0.7)',
                    borderColor: '#6366f1',
                    borderWidth: 1
                }}]
            }},
            options: commonOptions('Sample Count')
        }});
    </script>
</body>
</html>
"""
        with open(self.html_output_path, 'w') as f:
            f.write(html_content)

        # Suy ra đường dẫn tương ứng trên máy Host để user dễ mở trình duyệt
        host_html_path = self.html_output_path.replace('/sim_ws/src', '~/f1_ws/src')
        self.get_logger().info("==========================================================")
        self.get_logger().info(f" 🎉 SUCCESS! HTML Research Report generated:")
        self.get_logger().info(f" 📦 In Docker: {self.html_output_path}")
        self.get_logger().info(f" 💻 On Host:   {host_html_path}")
        self.get_logger().info("==========================================================")

    def destroy_node(self):
        try:
            self.csv_file.flush()
            self.csv_file.close()
            self.generate_html_report()
        except Exception as e:
            print(f"[eval_ai_performance] Error on shutdown report generation: {e}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AiPerformanceEvaluatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Stopping evaluation node & generating research report...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
