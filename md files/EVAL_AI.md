# F1TENTH AI Performance Evaluation & Benchmarking (`eval_ai_performance.py`)

Tài liệu này giải thích chi tiết cơ sở lý thuyết, định nghĩa **Ground Truth**, **Chuyên gia (Expert)**, các công thức toán học và quy trình vận hành node **`eval_ai_performance.py`** phục vụ công tác nghiên cứu khoa học và viết báo cáo/bài báo chuyên ngành.

---

## 1. Tổng quan & Mục đích Nghiên cứu

Trong nghiên cứu Robot tự hành và Học máy (Autonomous Racing & Imitation Learning), việc đánh giá mô hình không thể chỉ dừng lại ở nhận xét cảm tính *"xe chạy được"* hay *"không đâm tường"*. 

Node **`eval_ai_performance.py`** đóng vai trò là bộ công cụ **Đánh giá Định lượng (Quantitative Benchmarking)** và **Trực quan hóa Dữ liệu (Interactive Visualization)** chạy thời gian thực. Node giúp đo đạc chính xác mức độ hiệu quả, độ mượt và sai số bám đường của mô hình AI so với chuẩn lý thuyết.

---

## 2. Khái niệm Ground Truth & Expert (Trong Lý thuyết và Codebase)

### 2.1. Ground Truth (GT) Raceline là gì?
* **Khái niệm:** **Ground Truth (GT)** là đường đua mẫu lý tưởng hoàn hảo $(x_{\text{gt}}, y_{\text{gt}})$ đại diện cho quỹ đạo tối ưu mà xe cần tuân theo.
* **Vị trí trong Code:**
  * Được định nghĩa và tải từ file đường đua chuẩn `f1tenth_waypoint.csv` trong hàm `load_waypoints()` (dòng 115–127).
  * Toàn bộ các phép đo độ lệch không gian của xe đều lấy đường Ground Truth này làm gốc tọa độ tham chiếu.

### 2.2. Expert Pure Pursuit Controller (Chuyên gia) là gì?
* **Khái niệm:** **Expert** đại diện cho quyết định điều khiển tối ưu của bộ quy hoạch chuyên gia hình học tại **đúng vị trí và thời điểm** mà xe AI đang đứng.
* **Vị trí trong Code:**
  * Được tính toán tự động ở mỗi bước thời gian trong hàm `compute_expert_ground_truth()` (dòng 155–190).
  * Khi xe AI đang lái ở tọa độ thực $(x, y, \psi)$ do Odometry phát về, hàm này sẽ đóng vai trò chuyên gia để trả lời câu hỏi: *"Nếu tại vị trí $(x, y, \psi)$ này do Chuyên gia Pure Pursuit điều khiển thì góc lái chuẩn $\delta_{\text{Expert}}$ sẽ là bao nhiêu?"*.

### 2.3. AI Policy under Test
* **Khái niệm:** Là lệnh bẻ lái $\delta_{\text{AI}}$ và tốc độ $v_{\text{AI}}$ thực tế do mô hình AI (ONNX / PyTorch) tính toán và phát ra topic `/drive`.

---

## 3. Các Chỉ số Đánh giá Nghiên cứu (Research Metrics & Formulae)

Node tự động thu thập chuỗi thời gian và tính toán các chỉ số thống kê chuẩn trong các bài báo khoa học:

### 3.1. Cross-Track Error (CTE — Độ lệch làn thực tế)
Khoảng cách vuông góc/ngắn nhất từ vị trí xe $(x_{\text{car}}, y_{\text{car}})$ tới đường raceline chuẩn Ground Truth:
$$\text{CTE} = \min_{i} \sqrt{(x_{\text{car}} - x_{\text{gt}, i})^2 + (y_{\text{car}} - y_{\text{gt}, i})^2}$$

* **Mean CTE ($\overline{\text{CTE}}$):** Độ lệch làn trung bình trong suốt hành trình (mét).
* **Max CTE ($\text{CTE}_{\max}$):** Độ lệch làn cực đại tại khúc cua gắt nhất.

### 3.2. Steering Angle RMSE (Root Mean Squared Error)
Sai số căn bình phương trung bình giữa góc lái của AI ($\delta_{\text{AI}}$) và góc lái Chuyên gia ($\delta_{\text{Expert}}$):
$$\text{RMSE}_{\delta} = \sqrt{\frac{1}{N} \sum_{t=1}^{N} (\delta_{\text{AI}, t} - \delta_{\text{Expert}, t})^2}$$

* Thể hiện mức độ "lệch tay lái" tổng thể của AI so với chuyên gia (được tính theo cả Radian và Độ $^\circ$).

### 3.3. Steering Angle MAE (Mean Absolute Error)
Sai số tuyệt đối trung bình của góc lái:
$$\text{MAE}_{\delta} = \frac{1}{N} \sum_{t=1}^{N} \left| \delta_{\text{AI}, t} - \delta_{\text{Expert}, t} \right|$$

### 3.4. Vận tốc & Thời gian Hành trình
* **Mean Speed & Speed RMSE:** Đánh giá độ ổn định duy trì tốc độ của AI so với tốc độ chuyên gia đề xuất.
* **Duration:** Tổng thời gian thực hiện bài test và số lượng mẫu thu thập.

---

## 4. Đồ thị Trực quan & Báo cáo HTML (`ai_eval_report.html`)

Khi dừng node (`Ctrl + C`), hệ thống tự động tổng hợp toàn bộ log và sinh ra báo cáo nghiên cứu dạng web tương tác HTML (Chart.js) bao gồm 4 biểu đồ:

1. **Steering Angle Trajectory ($\delta_{\text{AI}}$ vs $\delta_{\text{Expert}}$):** Biểu đồ chuỗi thời gian so sánh chi tiết từng cú bẻ lái của AI với Chuyên gia.
2. **Cross-Track Error (CTE) over Time:** Biểu đồ thể hiện biến thiên độ lệch làn theo thời gian, giúp phát hiện các đoạn cua xe bị dạt lề.
3. **2D Spatial Trajectory Map:** Bản đồ không gian 2D mặt phẳng $(X, Y)$ đặt đường xe AI chạy chồng lên đường raceline chuẩn Ground Truth.
4. **Steering Error Distribution:** Biểu đồ tần suất (Histogram) phân bố sai số góc lái $(\delta_{\text{AI}} - \delta_{\text{Expert}})$.

---

## 5. Quy trình Vận hành Chi tiết

### Bước 1: Khởi động Mô phỏng & Chạy Xe bằng AI (Terminal 1)
Vào container Docker `f1tenth_sim` và chạy node suy luận AI:
```bash
docker exec -it f1tenth_sim bash
source install/setup.bash
ros2 run pure_pursuit_controller ai_inference_sim.py
# Hoặc chạy odom model:
# ros2 run pure_pursuit_controller ai_inference_sim_odom.py
```

### Bước 2: Chạy Node Đánh giá Nghiên cứu (Terminal 2)
Mở một Terminal Docker khác và khởi động node đánh giá:
```bash
docker exec -it f1tenth_sim bash
source install/setup.bash
ros2 run pure_pursuit_controller eval_ai_performance.py
```

### Bước 3: Nhận Kết quả & Mở Báo cáo trên Trình duyệt Host
* Để xe AI chạy tự lái 1 hoặc nhiều vòng.
* Nhấn `Ctrl + C` tại Terminal 2 để kết thúc đợt đánh giá.
* Mở trình duyệt web ngoài máy Host (Ubuntu Laptop) và mở file báo cáo:
```bash
google-chrome ~/f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/ai_eval_report.html
```

---

## 6. Sơ đồ Cấu trúc File Đầu ra

```
f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/
├── eval_ai_performance.py       # Code node ROS 2 đánh giá
├── ai_eval_log.csv              # Dữ liệu thô chuỗi thời gian (t, x, y, yaw, cte, steer_ai, steer_expert, ...)
└── ai_eval_report.html          # Báo cáo đồ thị tương tác HTML Sleek Dark Mode
```
