# Walkthrough: Architecture Phân Tán DAgger (Laptop & Jetson)

Chúng ta đã thiết lập và hoàn thiện thành công **Kiến Trúc DAgger Phân Tán (Distributed DAgger)** tách biệt công việc giữa Laptop và Jetson!

---

## 🛠️ Các File Mới Được Tạo

### 1. 💻 `rrt_expert_relay_node.py` (Chạy trên Máy Laptop)
* **Vị trí file:** [rrt_expert_relay_node.py](file:///home/adt/f1_ws/src/pure_pursuit_controller/rrt_planner/rrt_expert_relay_node.py)
* **Nguồn:** Copy giữ nguyên 100% thuật toán RRT* né vật cản từ `code_chay_reak.py`.
* **Tính năng mới:**
  - Publish góc bẻ lái Chuyên gia RRT* vào Topic Trung Gian: **`/rrt_expert_drive`**.
  - Lắng nghe tín hiệu status từ Jetson gửi về qua **`/dagger_status`** (`0`: RRT* Cứu Nét, `1`: AI tự lái).
  - Khi nhận status `0`, node **kích hoạt ghi dữ liệu DAgger CSV** (với Pre-roll 2s history) trực tiếp lên ổ cứng Laptop (`datasets/dagger_dataset_real.csv`).

### 2. 🚀 `ai_safe_controller_jetson.py` (Chạy trên Bo Mạch Jetson)
* **Vị trí file:** [ai_safe_controller_jetson.py](file:///home/adt/f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/controllers/ai_safe_controller_jetson.py)
* **Tính năng:**
  - Siêu nhẹ, chỉ chạy duy nhất mô hình AI PyTorch `.pth` từ LiDAR `/scan`.
  - Nhận góc bẻ lái RRT* từ Laptop qua topic trung gian `/rrt_expert_drive`.
  - So sánh chênh lệch góc lái `steer_diff = abs(ai_steer - rrt_steer)`:
    * Nếu **AI lái mượt** ➔ Phát lệnh AI lên `/drive` và gửi status `1` về Laptop.
    * Nếu **RRT* Cứu Nét** ➔ Phát lệnh RRT* lên `/drive` và gửi status `0` về Laptop.

---

## 🎮 Hướng Dẫn Sử Dụng Chi Tiết

### 💻 BƯỚC 1: Chạy Trên Máy LAPTOP
Chạy node RRT* Expert Relay để tính toán đường đi và lưu DAgger CSV:
```bash
ros2 run pure_pursuit_controller rrt_expert_relay_node.py --ros-args \
  -p waypoint_path:=/home/adt/f1_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv \
  -p dataset_path:=datasets/dagger_dataset_real.csv \
  -p pre_roll_secs:=2.0
```

### 🚀 BƯỚC 2: Chạy Trên Bo Mạch JETSON (Xe Thật)
Copy duy nhất file [ai_safe_controller_jetson.py](file:///home/adt/f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/controllers/ai_safe_controller_jetson.py) lên Jetson và chạy:
```bash
ros2 run pure_pursuit_controller ai_safe_controller_jetson.py --ros-args \
  -p model_path:=combine_1.pth \
  -p override_threshold:=0.15 \
  -p override_hold_secs:=1.0
```

---

## 📊 Luồng Hoạt Động Khi Chạy Thực Tế:

1. Khi AI trên Jetson bẻ lái bình thường ➔ Status = `1` ➔ Laptop không ghi data.
2. Ngay khi AI bẻ lái sai chệch đường ➔ Jetson phát lệnh RRT* cứu xe & phát Status = `0` về Laptop ➔ Laptop lập tức **rút 2.0 giây lịch sử trước đó (Pre-roll)** + **thu thập tiếp dữ liệu RRT* cứu nguy** ➔ Lưu thẳng vào file CSV trên Laptop!
