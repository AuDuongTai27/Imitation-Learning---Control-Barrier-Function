# Cheatsheet: Đổi 60 ↔ 90 Beams

Mỗi lần đổi số beams, chỉ cần thay **đúng các chỗ sau**. Không cần đụng vào bất kỳ logic nào khác.

---

## Tổng hợp — 1 bảng nhìn là xong

| # | File | Chỗ cần đổi | 60 beams | 90 beams |
|---|------|------------|----------|----------|
| 1 | `data_collection_sim.py` | `target_beams` parameter | `60` | `90` |
| 2 | `data_collection_combined_sim.py` | `target_beams` parameter | `60` | `90` |
| 3 | `train.py` | `default=` của `--input_dim` | `60` | `90` |
| 4 | `export_onnx.py` | `default=` của `--input_dim` | `60` | `90` |
| 5 | `ai_inference_sim.py` | `target_beams` parameter | `60` | `90` |

> [!CAUTION]
> 5 chỗ trên phải **cùng một giá trị**. Lệch bất kỳ 1 chỗ là model nhận input sai kích thước → crash hoặc ra kết quả ngẫu nhiên.

---

## Chi tiết từng file

### 1. `data_collection_sim.py` — Thu thập dữ liệu LiDAR

```python
# Dòng 31
self.declare_parameter('target_beams', 60)   # ← đổi thành 90 nếu cần
```

Hoặc không sửa code, truyền qua tham số lúc chạy:
```bash
ros2 run pure_pursuit_controller data_collection_sim.py \
  --ros-args -p target_beams:=90
```

---

### 2. `data_collection_combined_sim.py` — Thu thập LiDAR + Odom

```python
self.declare_parameter('target_beams', 60)   # ← đổi thành 90 nếu cần
```

Hoặc qua tham số:
```bash
ros2 run pure_pursuit_controller data_collection_combined_sim.py \
  --ros-args -p target_beams:=90
```

---

### 3. `train.py` — Huấn luyện

Có **2 chỗ** cần chú ý trong file này:

```python
# Chỗ 1 — default của argparse (dòng cuối file, trong if __name__ == '__main__')
parser.add_argument('--input_dim', type=int, default=90,   # ← đổi default thành 60 nếu cần
                    help='Số cột LiDAR trong CSV ...')

# Chỗ 2 — default của --csv và --model (nên đổi tên file cho khớp)
parser.add_argument('--csv',   type=str, default='dagger_dataset_sim_5.csv')
parser.add_argument('--model', type=str, default='model_sim_5.pth')
```

Hoặc không sửa code, truyền tất cả qua CLI:
```bash
# 60 beams
python3 train.py --csv dagger_dataset_sim_4.csv --model my_model_60.pth --input_dim 60

# 90 beams
python3 train.py --csv dagger_dataset_sim_90.csv --model model_sim_90.pth --input_dim 90
```

> [!WARNING]
> `train.py` tự kiểm tra: nếu `--input_dim` không khớp số cột LiDAR thực tế trong CSV header → raise `ValueError` ngay lập tức trước khi train. Thông báo lỗi sẽ cho biết con số đúng cần truyền.

---

### 4. `export_onnx.py` — Export sang ONNX

```python
# Trong if __name__ == '__main__'
parser.add_argument('--input_dim', type=int, default=90,   # ← đổi thành 60 nếu cần
                    help='Số chiều input (số beam LiDAR)...')

# Nên đổi tên file output cho khớp
parser.add_argument('--model', type=str, default='model_sim_5.pth')
parser.add_argument('--onnx',  type=str, default='model_sim_5.onnx')
```

Hoặc truyền qua CLI:
```bash
# 60 beams
python3 export_onnx.py --model my_model_60.pth --onnx my_model_60.onnx --input_dim 60

# 90 beams
python3 export_onnx.py --model model_sim_90.pth --onnx model_sim_90.onnx --input_dim 90
```

> [!NOTE]
> `export_onnx.py` tự chạy **sanity check**: so sánh output PyTorch vs ONNX trên dummy input. Nếu `max_diff > 1e-4` sẽ cảnh báo — thường là dấu hiệu `--input_dim` bị truyền sai.

---

### 5. `ai_inference_sim.py` — Chạy AI trong sim

```python
# Dòng 39-40
self.declare_parameter('model_path', '.../dagger_model_sim_3.onnx')  # ← đổi tên file .onnx
self.declare_parameter('target_beams', 60)                            # ← đổi thành 90 nếu cần
```

Hoặc qua tham số (khuyến nghị — không cần rebuild):
```bash
ros2 run pure_pursuit_controller ai_inference_sim.py \
  --ros-args \
  -p model_path:=/sim_ws/.../model_sim_90.onnx \
  -p target_beams:=90
```

---

## Kiểm tra nhanh số beams của 1 file CSV

```bash
# Đếm tổng số cột, rồi trừ 2 (speed + steering_angle) = input_dim cần dùng
head -1 dagger_dataset_sim.csv | tr ',' '\n' | wc -l
# Kết quả 62  →  input_dim = 60
# Kết quả 92  →  input_dim = 90
```

---

## Luồng đầy đủ một lần chạy (copy & paste)

```bash
# ── 60 BEAMS ────────────────────────────────────────────────────
# 1. Thu thập (Docker)
ros2 run pure_pursuit_controller data_collection_sim.py \
  --ros-args -p target_beams:=60 -p dataset_path:=.../my60.csv

# 2. Train (local/Colab)
python3 train.py --csv my60.csv --model model60.pth --input_dim 60

# 3. Export ONNX
python3 export_onnx.py --model model60.pth --onnx model60.onnx --input_dim 60

# 4. Inference (Docker)
ros2 run pure_pursuit_controller ai_inference_sim.py \
  --ros-args -p model_path:=.../model60.onnx -p target_beams:=60

# ── 90 BEAMS ────────────────────────────────────────────────────
# 1. Thu thập (Docker)
ros2 run pure_pursuit_controller data_collection_sim.py \
  --ros-args -p target_beams:=90 -p dataset_path:=.../my90.csv

# 2. Train (local/Colab)
python3 train.py --csv my90.csv --model model90.pth --input_dim 90

# 3. Export ONNX
python3 export_onnx.py --model model90.pth --onnx model90.onnx --input_dim 90

# 4. Inference (Docker)
ros2 run pure_pursuit_controller ai_inference_sim.py \
  --ros-args -p model_path:=.../model90.onnx -p target_beams:=90
```
