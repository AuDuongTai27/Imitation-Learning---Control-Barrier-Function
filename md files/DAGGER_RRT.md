# DAgger với RRT Expert + Auto Obstacle Spawning (Phiên bản Cải tiến)

Hướng dẫn thu thập dữ liệu và huấn luyện AI né vật cản, hỗ trợ chống học thuộc lòng hướng né và tự động trả lái về đường (Recovery Phase).

---

## Các Lệnh Chạy Vận Hành Hệ Thống

### 1. Build lại package (Trong Docker)
```bash
docker exec -it f1tenth_sim bash
cd /sim_ws && colcon build --packages-select pure_pursuit_controller && source install/setup.bash
```

---

### 2. Chạy Auto Obstacle Spawner (Tự động tạo vật cản Trái / Giữa / Phải linh hoạt)
Node tự động tạo vật cản ngay khi xe vừa vượt qua vật cản cũ, luân phiên lệch Trái/Giữa/Phải với khoảng cách ngẫu nhiên `3.0m - 6.0m`:

```bash
ros2 run pure_pursuit_controller auto_obstacle_spawner.py \
  --ros-args \
  -p spawn_distance_ahead:=4.0 \
  -p max_perpendicular_offset:=0.22 \
  -p obstacle_radius:=0.28
```

---

### 3. Chạy DAgger RRT Collector (Thu thập Data Né + Data Trả Lái Recovery)
Ghi data khi phát hiện nguy hiểm và tự động ghi thêm 1.5 giây sau khi hết nguy hiểm để bắt trọn khoảnh khắc RRT bẻ lái đưa xe về raceline:

```bash
ros2 run pure_pursuit_controller dagger_rrt_collector.py \
  --ros-args \
  -p danger_threshold:=2.0 \
  -p danger_angle_deg:=30.0 \
  -p target_beams:=60 \
  -p dataset_path:=/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/dagger_rrt_dataset.csv
```

---

### 4. Gộp Dataset (Pure Pursuit raceline chuẩn + DAgger RRT né/trả lái)
Để mô hình không quên cách chạy bám đường phẳng khi không có vật cản:

```bash
# Tạo file dataset gộp
head -1 dagger_dataset_sim_4.csv > combined_rrt_dataset.csv
tail -n +2 dagger_dataset_sim_4.csv >> combined_rrt_dataset.csv   # Data chạy raceline chuẩn
tail -n +2 dagger_rrt_dataset.csv >> combined_rrt_dataset.csv      # Data DAgger RRT né & trả lái
```

---

### 5. Huấn luyện Model PyTorch (Chống Early Stop sớm)
Train với `--patience 25` và `--lr 0.0005` giúp Loss giảm mịn và sâu hơn:

```bash
python3 train.py \
  --csv combined_rrt_dataset.csv \
  --model rrt_model_v2.pth \
  --input_dim 60 \
  --epochs 150 \
  --batch_size 64 \
  --patience 25 \
  --lr 0.0005
```

---

### 6. Export sang ONNX để chạy Sim / Xe thật
```bash
python3 export_onnx.py \
  --model rrt_model_v2.pth \
  --onnx rrt_model_v2.onnx \
  --input_dim 60
```
