#!/usr/bin/env python3
"""
visualize_steering.py
─────────────────────
Đọc file `steering_log.csv` và sinh ra một báo cáo phân tích tương tác dưới dạng HTML 
(steering_analysis.html) sử dụng Chart.js để hiển thị dữ liệu góc lái và phản hồi odom.

Giải pháp này hoàn toàn loại bỏ phụ thuộc vào numpy và matplotlib, tránh được lỗi 
mismatch phiên bản NumPy trên Jetson/Host Laptop.
"""

import os
import csv
import json

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'steering_log.csv')
    html_path = os.path.join(current_dir, 'steering_analysis.html')
    
    if not os.path.exists(csv_path):
        print(f"❌ File log '{csv_path}' không tồn tại. Hãy chạy steering_logger.py trước để thu thập dữ liệu.")
        return
        
    print(f"📖 Đang đọc dữ liệu từ {csv_path}...")
    
    drive_data = [] # List of dict: {"x": time_s, "y": steer, "speed": speed}
    odom_data = []  # List of dict: {"x": time_s, "y": yaw_rate, "speed": speed}
    
    first_timestamp = None
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None) # Bỏ qua header
            
            for row in reader:
                if not row or len(row) < 4:
                    continue
                try:
                    t_ns = int(row[0])
                    topic = row[1]
                    val1 = float(row[2])
                    val2 = float(row[3])
                    
                    if first_timestamp is None:
                        first_timestamp = t_ns
                        
                    time_s = round((t_ns - first_timestamp) / 1e9, 4) # Đổi sang giây
                    
                    if topic == 'drive':
                        # val1: steering_angle (rad), val2: speed (m/s)
                        drive_data.append({"x": time_s, "y": val1, "speed": val2})
                    elif topic == 'odom':
                        # val1: linear_x (m/s), val2: angular_z (rad/s)
                        odom_data.append({"x": time_s, "y": val2, "speed": val1})
                except ValueError:
                    continue
    except Exception as e:
        print(f"❌ Lỗi khi đọc file CSV: {e}")
        return

    if not drive_data and not odom_data:
        print("❌ Không có dữ liệu hợp lệ trong file CSV.")
        return

    print(f"📊 Đã tải: {len(drive_data)} điểm Lệnh lái (drive), {len(odom_data)} điểm Phản hồi (odom).")

    # Tính toán một số thống kê nhanh
    steer_vals = [d["y"] for d in drive_data]
    yaw_rate_vals = [d["y"] for d in odom_data]
    max_steer = max(steer_vals) if steer_vals else 0
    min_steer = min(steer_vals) if steer_vals else 0
    max_yaw_rate = max(yaw_rate_vals) if yaw_rate_vals else 0
    min_yaw_rate = min(yaw_rate_vals) if yaw_rate_vals else 0
    
    print(f"   - Góc lái lệnh lớn nhất: {max_steer:.3f} rad ({max_steer * 57.2958:.1f} độ)")
    print(f"   - Góc lái lệnh nhỏ nhất: {min_steer:.3f} rad ({min_steer * 57.2958:.1f} độ)")
    print(f"   - Tốc độ xoay odom lớn nhất (Yaw rate): {max_yaw_rate:.3f} rad/s")

    # Tạo nội dung HTML chứa Chart.js
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>F1TENTH Steering & Latency Analysis</title>
    <!-- CSS Sleek Dark Mode -->
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #121214;
            color: #e2e8f0;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #6366f1;
            margin-bottom: 5px;
            font-weight: 600;
        }}
        .subtitle {{
            text-align: center;
            color: #94a3b8;
            margin-bottom: 30px;
        }}
        .card {{
            background: rgba(30, 41, 59, 0.7);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-box {{
            background: #1e1b4b;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #6366f1;
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #818cf8;
            margin-top: 5px;
        }}
        .stat-label {{
            font-size: 12px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .chart-container {{
            position: relative;
            height: 320px;
            width: 100%;
        }}
        .tip-box {{
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.2);
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        .tip-title {{
            font-weight: bold;
            color: #818cf8;
            margin-bottom: 5px;
        }}
    </style>
    <!-- Load Chart.js từ CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <h1>📊 BÁO CÁO PHÂN TÍCH ĐỘ TRỄ LÁI XE F1TENTH</h1>
        <div class="subtitle">Dữ liệu được trích xuất tự động từ steering_log.csv</div>

        <!-- Bảng thống kê tóm tắt -->
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-label">Tổng số điểm ghi nhận</div>
                <div class="stat-value">{len(drive_data) + len(odom_data)}</div>
            </div>
            <div class="stat-box" style="border-left-color: #10b981;">
                <div class="stat-label">Góc lái lệnh cực đại</div>
                <div class="stat-value">{max_steer:.3f} rad ({max_steer * 57.3:.1f}°)</div>
            </div>
            <div class="stat-box" style="border-left-color: #3b82f6;">
                <div class="stat-label">Tốc độ xoay thực tế cực đại</div>
                <div class="stat-value">{max_yaw_rate:.3f} rad/s</div>
            </div>
            <div class="stat-box" style="border-left-color: #f59e0b;">
                <div class="stat-label">Thời gian ghi log</div>
                <div class="stat-value">{max(drive_data[-1]["x"] if drive_data else 0, odom_data[-1]["x"] if odom_data else 0):.1f} s</div>
            </div>
        </div>

        <!-- Biểu đồ 1: Lệnh góc lái -->
        <div class="card">
            <h3>1. Lệnh Góc Lái Gửi Đi (Commanded Steering Angle)</h3>
            <div class="chart-container">
                <canvas id="steerChart"></canvas>
            </div>
        </div>

        <!-- Biểu đồ 2: Tốc độ xoay odom -->
        <div class="card">
            <h3>2. Tốc Độ Xoay Thực Tế Của Xe (Actual Yaw Rate)</h3>
            <div class="chart-container">
                <canvas id="yawChart"></canvas>
            </div>
        </div>

        <!-- Biểu đồ 3: So sánh Tốc độ -->
        <div class="card">
            <h3>3. So Sánh Tốc Độ: Lệnh Gửi Đi vs Thực Tế Từ Odom</h3>
            <div class="chart-container">
                <canvas id="speedChart"></canvas>
            </div>
        </div>

        <!-- Hướng dẫn phân tích -->
        <div class="tip-box">
            <div class="tip-title">💡 HƯỚNG DẪN ĐÁNH GIÁ ĐỘ TRỄ LÁI (LATENCY ANALYSIS):</div>
            Làm sao để biết xe bị trễ lái vật lý?<br>
            1. Cuộn chuột để zoom hoặc hover chuột vào các đỉnh/thung lũng nhấp nhô của góc lái trên <b>Biểu đồ 1</b>.<br>
            2. Đối chiếu thời gian (trục X) của đỉnh đó với đỉnh phản hồi tương ứng trên <b>Biểu đồ 2 (Yaw Rate)</b>.<br>
            3. <b>Công thức tính trễ:</b> Độ trễ = [Thời gian đạt đỉnh Yaw Rate] - [Thời gian đạt đỉnh Lệnh lái].<br>
            Nếu con số này lớn hơn <b>0.15 giây</b> (150ms), xe thật của bạn đang bị trễ phản ứng servo cơ học đáng kể, khiến xe khó ôm cua kịp thời ở tốc độ cao!
        </div>
    </div>

    <script>
        // Nhúng trực tiếp dữ liệu từ file log vào JS
        const driveData = {json.dumps(drive_data)};
        const odomData = {json.dumps(odom_data)};

        // Cấu hình chung cho Chart.js
        const commonOptions = (yLabel) => ({{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
                x: {{
                    type: 'linear',
                    position: 'bottom',
                    title: {{
                        display: true,
                        text: 'Thời gian (giây - seconds)',
                        color: '#94a3b8'
                    }},
                    grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                    ticks: {{ color: '#94a3b8' }}
                }},
                y: {{
                    title: {{
                        display: true,
                        text: yLabel,
                        color: '#94a3b8'
                    }},
                    grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                    ticks: {{ color: '#94a3b8' }}
                }}
            }},
            plugins: {{
                legend: {{
                    labels: {{ color: '#e2e8f0' }}
                }}
            }}
        }});

        // Vẽ Steer Chart
        new Chart(document.getElementById('steerChart'), {{
            type: 'line',
            data: {{
                datasets: [{{
                    label: 'Commanded Steer (rad)',
                    data: driveData,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    showLine: true
                }}]
            }},
            options: commonOptions('Góc lái (rad)')
        }});

        // Vẽ Yaw Chart
        new Chart(document.getElementById('yawChart'), {{
            type: 'line',
            data: {{
                datasets: [{{
                    label: 'Actual Yaw Rate (rad/s)',
                    data: odomData,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    showLine: true
                }}]
            }},
            options: commonOptions('Yaw Rate (rad/s)')
        }});

        // Vẽ Speed Chart
        new Chart(document.getElementById('speedChart'), {{
            type: 'line',
            data: {{
                datasets: [
                    {{
                        label: 'Cmd Speed (m/s)',
                        data: driveData.map(d => ({{x: d.x, y: d.speed}})),
                        borderColor: '#f59e0b',
                        borderWidth: 1.5,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        showLine: true
                    }},
                    {{
                        label: 'Act Speed (m/s)',
                        data: odomData.map(o => ({{x: o.x, y: o.speed}})),
                        borderColor: '#ec4899',
                        borderWidth: 2,
                        pointRadius: 0,
                        showLine: true
                    }}
                ]
            }},
            options: commonOptions('Vận tốc (m/s)')
        }});
    </script>
</body>
</html>
"""

    try:
        with open(html_path, 'w') as f:
            f.write(html_content)
        print(f"🎉 Biểu đồ phân tích độ trễ dạng tương tác đã được lưu thành công tại:\n   👉 {html_path}")
        print("💡 Bạn chỉ cần mở file HTML này bằng bất kỳ trình duyệt nào để xem đồ thị tương tác cực kỳ mượt mà!")
    except Exception as e:
        print(f"❌ Không thể ghi file HTML: {e}")

if __name__ == '__main__':
    main()
