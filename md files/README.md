# F1TENTH Autonomous Simulation & Expert Planning Workspace (`f1_ws`)

Tài liệu này hướng dẫn chi tiết cấu trúc thư mục, các tệp tin quan trọng và quy trình từng bước để vận hành hệ thống mô phỏng F1TENTH, thu thập dữ liệu (DAgger), chạy bộ lập quỹ đạo chuyên gia (RRT* / Pure Pursuit) và chạy xe tự hành bằng AI trong môi trường Docker Container.

---

## 1. Bản đồ Đường dẫn & Các Tệp tin Quan trọng

Workspace `f1_ws` bao gồm 3 gói ROS 2 chính đặt trong thư mục `src/`:
1.  **`f1tenth_gym_ros`**: Cầu nối mô phỏng và môi trường Gym (ROS 2 Bridge).
2.  **`waypoint`**: Bộ sinh đường đua tĩnh và phát (publish) danh sách điểm mốc (waypoints).
3.  **`pure_pursuit_controller`**: Chứa thuật toán điều khiển Pure Pursuit, thuật toán quy hoạch đường đi RRT, thu thập dữ liệu DAgger, huấn luyện PyTorch và chạy suy luận AI (Inference).

### Các tệp tin quan trọng nhất trong gói `pure_pursuit_controller`:

| Tên File | Chức năng / Ý nghĩa |
| :--- | :--- |
| **`rrt_planner/rrt.py`** | Lớp lõi thuật toán RRT (đầy đủ chức năng lọc va chạm, cắt tỉa đường đi `pruning` và làm mịn góc `B-spline`). |
| **`rrt_planner/hybrid_planner_sim.py`** | Node ROS 2 RRT chuyên gia chạy trong **mô phỏng** (đã loại bỏ chế độ lùi xe khi không cần thiết, tự động hóa frame `ego_racecar/base_link`). |
| **`rrt_planner/code_chay_reak.py`** | Node RRT chuyên gia chạy trên **xe thật** (sử dụng frame `"base_link"` và cấu hình thực tế). |
| **`pure_pursuit_controller/controller_simulation.py`** | Node điều khiển bám đường tĩnh Pure Pursuit tiêu chuẩn trong mô phỏng. |
| **`pure_pursuit_controller/controller_real_run.py`** | Node điều khiển bám đường tĩnh Pure Pursuit chạy trên xe thật (hỗ trợ TF lookup map $\rightarrow$ base_link và fallback odom). |
| **`pure_pursuit_controller/data_collection_sim.py`** | Node thu thập dữ liệu chuyên gia chạy thường (Behavioral Cloning) trong mô phỏng (ghi đè LiDAR + lệnh lái của RRT/Pure Pursuit khi họ lái). |
| **`pure_pursuit_controller/data_collection_real.py`** | Node thu thập dữ liệu chuyên gia chạy thường (Behavioral Cloning) trên xe thật. |
| **`pure_pursuit_controller/train.py`** | Script Python huấn luyện mạng nơ-ron (MLP) bám đường tránh vật cản bằng PyTorch (Local/Offline). |
| **`pure_pursuit_controller/ai_inference_sim.py`** | Node chạy xe tự hành AI tự lái thuần túy (Autonomous Only) bằng ONNX Runtime trong mô phỏng. |
| **`pure_pursuit_controller/ai_inference_sim_override.py`** | Node chạy AI tự lái kết hợp cơ chế tự động ghi đè (Override) của chuyên gia Pure Pursuit khi lệch làn để thu thập dữ liệu **DAgger** trong mô phỏng. |
| **`pure_pursuit_controller/ai_inference_real.py`** | Node chạy xe tự hành AI tự lái thuần túy bằng ONNX Runtime trên xe thật Jetson. |
| **`pure_pursuit_controller/ai_inference_real_pytorch.py`** | Node chạy xe tự hành AI tự lái thuần túy bằng PyTorch (sử dụng trực tiếp file `.pth`) trên xe thật Jetson (ổn định cao). |
| **`pure_pursuit_controller/eval_ai_performance.py`** | Node đánh giá định lượng hiệu suất AI (Steering RMSE, MAE, Cross-Track Error) & xuất báo cáo đồ thị HTML tương tác ([Xem chi tiết EVAL_AI.md](EVAL_AI.md)). |
| **`pure_pursuit_controller/obstacle_spawner.py`** | Node tùy chọn hỗ trợ sinh vật cản ngẫu nhiên (tĩnh/động) trước đầu xe để thử nghiệm tránh va chạm. |

---

## 2. Hướng dẫn Chạy tuần tự hệ thống (Copy & Paste)

### Bước chuẩn bị (Máy Host)
Đảm bảo thư mục workspace `f1_ws` đã được đặt ngay tại thư mục Home của máy host:
```bash
# Đường dẫn chuẩn phải là:
~/f1_ws
```

---

### Bước 1: Khởi động Docker Container (Terminal 1)
Mở terminal đầu tiên trên máy Host, cấp quyền hiển thị giao diện đồ họa X11 (RViz2) cho Docker và chạy docker compose:
```bash
cd ~/f1_ws/src/f1tenth_gym_ros
bash launch.sh
```
*(Giữ nguyên Terminal này để theo dõi tiến trình của Docker)*.

---

### Bước 2: Chạy Cầu nối Mô phỏng (Terminal 2 - Docker)
Mở một Terminal mới trên máy Host, chạy lệnh để đi vào Docker container:
```bash
docker exec -it f1tenth_sim bash
```
Sau đó, copy khối lệnh sau chạy bên trong Docker để build, source và khởi động simulator:
```bash
colcon build
source install/setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```
*(Giao diện mô phỏng 2D và RViz2 sẽ hiện lên)*.

---

### Bước 3: Phát danh sách Waypoint (Terminal 3 - Docker)
Mở một Terminal mới trên máy Host, chạy lệnh để đi vào Docker container:
```bash
docker exec -it f1tenth_sim bash
```
Sau đó, copy khối lệnh sau chạy bên trong Docker để khởi động node waypoint:
```bash
colcon build
source install/setup.bash
ros2 run waypoint waypoint_generator_node.py
```
> [!IMPORTANT]
> **Sau khi chạy node này:** Bạn cần mở cửa sổ **RViz2** lên, sử dụng công cụ **Publish Point** (trên thanh công cụ phía trên cùng của RViz2) để click liên tiếp các điểm mốc trên bản đồ nhằm tự vẽ/sinh đường đua mong muốn.

---

### Bước 4: Khởi chạy Bộ quy hoạch Chuyên gia RRT (Terminal 4 - Docker)
Mở một Terminal mới trên máy Host, chạy lệnh để đi vào Docker container:
```bash
docker exec -it f1tenth_sim bash
```
Sau đó, copy khối lệnh sau chạy bên trong Docker để khởi chạy chuyên gia RRT:
```bash
colcon build
source install/setup.bash
ros2 run pure_pursuit_controller hybrid_planner_sim.py
```
*(Lúc này xe trên mô phỏng sẽ bắt đầu tự lái bám theo đường chạy RRT sinh ra)*.

---

### Bước 5: Thu thập Dữ liệu huấn luyện (Terminal 5 - Docker)
Mở một Terminal mới trên máy Host, chạy lệnh để đi vào Docker container:
```bash
docker exec -it f1tenth_sim bash
```
Sau đó, copy khối lệnh sau chạy bên trong Docker để khởi chạy ghi dữ liệu:
```bash
colcon build
source install/setup.bash
ros2 run pure_pursuit_controller data_collection_sim.py
```
*(Dữ liệu sẽ được lưu tự động thành các file `.csv` trong gói `pure_pursuit_controller`)*.

---

### Bước 6: Sinh vật cản để thử thách xe (Terminal 6 - Docker - Tùy chọn)
Nếu bạn muốn tạo thêm vật cản trên đường chạy để kiểm tra khả năng né tránh của RRT và thu thập dữ liệu né vật cản:

> [!WARNING]
> **Khuyến nghị quan trọng:** Vì node sinh vật cản này sử dụng chung công cụ **Publish Point** của RViz2 giống như node phát waypoint ở **Bước 3**, bạn nên **tắt node Waypoint ở Terminal 3** (nhấn `Ctrl + C` để dừng) trước khi chạy node này. Tránh tình trạng mỗi khi click trên bản đồ, cả 2 node đều nhận tín hiệu khiến vừa tạo waypoint vừa sinh vật cản chồng chéo lên nhau.

Mở một Terminal mới trên máy Host, chạy lệnh để đi vào Docker container:
```bash
docker exec -it f1tenth_sim bash
```
Sau đó, copy khối lệnh sau chạy bên trong Docker để sinh vật cản:
```bash
colcon build
source install/setup.bash
ros2 run pure_pursuit_controller obstacle_spawner.py
```

---

## 3. Huấn luyện & Chạy xe bằng AI

Sau khi thu thập đủ dữ liệu `.csv`, bạn có thể tiến hành huấn luyện mạng nơ-ron và cho xe tự chạy bằng AI:

### Bước A: Huấn luyện Mô hình AI
Đem tệp tin dữ liệu `.csv` đã thu thập được tải lên notebook Google Colab sau để huấn luyện:
👉 **[Google Colab - Train F1TENTH DAgger Model](https://colab.research.google.com/drive/1uz1E_ttox1ylTNWbfkaQ2pwItH739ivk?usp=sharing)**

*(Sau khi train xong trên Colab, bạn tải file mô hình đã chuyển đổi `.onnx` hoặc `.pth` về, đặt vào thư mục `pure_pursuit_controller/pure_pursuit_controller/`)*.

### Bước B: Cho xe chạy tự động bằng AI (Trong Docker)
Tắt node RRT Chuyên gia ở **Terminal 4**, chạy lệnh sau để vào Docker container:
```bash
docker exec -it f1tenth_sim bash
```
Sau đó, copy khối lệnh sau chạy bên trong Docker để chạy xe tự lái bằng AI:
```bash
colcon build
source install/setup.bash
ros2 run pure_pursuit_controller ai_inference_sim.py
```

---

## 4. Danh sách các Topic trực quan hóa trên RViz2 (Visualization Topics)

Để theo dõi trực quan quỹ đạo, mô hình lập trình và các vật cản trên giao diện đồ họa **RViz2**, bạn hãy nhấn **Add -> By Topic** ở phía dưới bên trái màn hình RViz2 và thêm các topic quan trọng sau:

1. **Đường đua tĩnh / Waypoints:**
   *   Topic: `/f1tenth_waypoint_marker` (Kiểu: `Marker`) $\rightarrow$ Hiển thị danh sách các điểm mốc và đường đua do bạn vẽ thủ công hoặc load từ file.
2. **Bộ quy hoạch RRT* (Local Map & Path):**
   *   Topic: `/visualization/markers` (Kiểu: `MarkerArray`) $\rightarrow$ Hiển thị cây tìm kiếm RRT (RRT Tree) và đường đi đề xuất (màu xanh lá/màu đỏ) bám sát xe.
   *   Topic: `/local_map_debug` (Kiểu: `OccupancyGrid`) $\rightarrow$ Hiển thị lưới bản đồ cục bộ xung quanh xe (thể hiện rõ vùng đen đã giãn nở an toàn bao quanh vật cản).
3. **Vật cản ảo mô phỏng:**
   *   Topic: `/sim_obstacle` (Kiểu: `Marker`) $\rightarrow$ Hiển thị chướng ngại vật màu đỏ được tạo ra khi bạn click chuột bằng công cụ "Publish Point".

---

