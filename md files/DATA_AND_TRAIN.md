# Hướng dẫn Thu thập Dữ liệu & Huấn luyện AI — F1TENTH Simulation

Tài liệu này giải thích toàn bộ pipeline từ **thu thập dữ liệu** → **huấn luyện** → **inference** cho hệ thống AI F1TENTH, bao gồm cả hai nhánh: **LiDAR-based** và **Odometry-based**.

---

## Mục lục

1. [Tổng quan Pipeline](#1-tổng-quan-pipeline)
2. [Các File Thu thập Dữ liệu](#2-các-file-thu-thập-dữ-liệu)
   - [data_collection_sim.py — LiDAR](#21-data_collection_simpy--lidar)
   - [data_collection_odom_sim.py — Odometry](#22-data_collection_odom_simpy--odometry)
   - [data_collection_combined_sim.py — LiDAR + Odom](#23-data_collection_combined_simpy--lidar--odom-kết-hợp)
3. [Format CSV & Ý nghĩa từng Cột](#3-format-csv--ý-nghĩa-từng-cột)
4. [File Huấn luyện](#4-file-huấn-luyện)
   - [train.py — LiDAR Model](#41-trainpy--lidar-model)
   - [Tham số `--input_dim`: Cốt lõi cần hiểu](#42-tham-số---input_dim-cốt-lõi-cần-hiểu)
   - [train_odom.py — Odometry Model](#43-train_odomp--odometry-model)
5. [File Inference (Chạy AI trên Sim)](#5-file-inference-chạy-ai-trên-sim)
6. [Normalization & File `_norm.json`](#6-normalization--file-_normjson)
7. [Workflow từ đầu đến cuối](#7-workflow-từ-đầu-đến-cuối)
8. [Bảng tra cứu nhanh: Phối hợp Data — Train — Inference](#8-bảng-tra-cứu-nhanh-phối-hợp-data--train--inference)

---

## 1. Tổng quan Pipeline

```
[Sim chạy RRT/Pure Pursuit]
         │
         ▼
[Thu thập dữ liệu] ──→ CSV
         │
         ▼
[train.py / train_odom.py]  (trên máy local hoặc Colab)
         │
         ▼
[model.pth]  ──export──→  [model.onnx + model_norm.json]
         │
         ▼
[ai_inference_sim.py / ai_inference_sim_odom.py]  (trong Docker)
```

Có **hai nhánh song song** hoàn toàn độc lập:

| Nhánh | Input Model | Script Thu thập | Script Train | Script Inference |
|-------|-------------|-----------------|--------------|-----------------|
| **LiDAR** | 60 hoặc 90 beams LiDAR | `data_collection_sim.py` | `train.py` | `ai_inference_sim.py` |
| **Odometry** | 6 chiều (x, y, yaw, vx, vy, wz) | `data_collection_odom_sim.py` | `train_odom.py` | `ai_inference_sim_odom.py` |
| **Kết hợp** | 73 chiều (60 LiDAR + 11 odom) | `data_collection_combined_sim.py` | `train.py --input_dim 73` _(xem mục 4.2)_ | tùy chỉnh |

---

## 2. Các File Thu thập Dữ liệu

> **Cách dùng chung cho tất cả file thu thập:**
> Chạy song song với expert node (RRT/Pure Pursuit). Các node này **chỉ lắng nghe**, không điều khiển xe.

### 2.1 `data_collection_sim.py` — LiDAR

**Subscribe:** `/scan` (LiDAR), `/drive` (lệnh lái chuyên gia)

```bash
# Trong Docker, sau colcon build && source install/setup.bash:
ros2 run pure_pursuit_controller data_collection_sim.py

# Tùy chỉnh số beam và đường lưu file:
ros2 run pure_pursuit_controller data_collection_sim.py \
  --ros-args \
  -p dataset_path:=/sim_ws/.../my_data.csv \
  -p target_beams:=60 \
  -p max_range:=10.0 \
  -p buffer_size:=50
```

**Tham số quan trọng:**

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `dataset_path` | `dagger_dataset_sim_5.csv` | Đường dẫn file CSV output |
| `target_beams` | `60` | Số tia LiDAR sau downsample |
| `max_range` | `10.0` | Giá trị thay thế cho NaN/Inf (mét) |
| `buffer_size` | `50` | Số dòng gộp lại trước khi ghi file (I/O performance) |

**Cơ chế đồng bộ:**
- Trigger: mỗi lần nhận `/scan` mới
- Kiểm tra `/drive` có đến trong vòng **0.5 giây** không → nếu không, bỏ qua sample
- LiDAR được **crop [-60°, +60°]** rồi **nội suy** xuống `target_beams` điểm

---

### 2.2 `data_collection_odom_sim.py` — Odometry

**Subscribe:** `/ego_racecar/odom` (Odometry), `/drive` (lệnh lái chuyên gia)

```bash
ros2 run pure_pursuit_controller data_collection_odom_sim.py

# Tùy chỉnh:
ros2 run pure_pursuit_controller data_collection_odom_sim.py \
  --ros-args \
  -p dataset_path:=/sim_ws/.../odom_data.csv \
  -p odom_topic:=/ego_racecar/odom \
  -p drive_topic:=/drive \
  -p buffer_size:=50
```

**Trigger:** mỗi lần nhận `/ego_racecar/odom` mới, kiểm tra `/drive` trong vòng **0.5 giây**.

> [!WARNING]
> Model Odometry chỉ hoạt động đúng khi xe khởi động **đúng vị trí / hướng** đã thu thập, và odometry **không bị drift**. Đây là model "học thuộc lòng quỹ đạo", không phải model nhận thức môi trường.

---

### 2.3 `data_collection_combined_sim.py` — LiDAR + Odom (Kết hợp)

**Subscribe:** `/scan`, `/ego_racecar/odom`, `/drive`

```bash
ros2 run pure_pursuit_controller data_collection_combined_sim.py

# Tùy chỉnh:
ros2 run pure_pursuit_controller data_collection_combined_sim.py \
  --ros-args \
  -p dataset_path:=/sim_ws/.../combined_data.csv \
  -p target_beams:=60 \
  -p odom_timeout:=0.1 \
  -p drive_timeout:=0.5
```

**Tham số bổ sung so với bản LiDAR:**

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `odom_topic` | `/ego_racecar/odom` | Topic Odometry |
| `odom_timeout` | `0.1` giây | Ngưỡng đồng bộ odom (chặt hơn drive vì odom publish nhanh hơn) |
| `drive_timeout` | `0.5` giây | Ngưỡng đồng bộ drive |

**Trigger:** `/scan` → snapshot `latest_odom` (0.1s) + `latest_drive` (0.5s) → ghi 1 hàng CSV.

---

## 3. Format CSV & Ý nghĩa từng Cột

### CSV từ `data_collection_sim.py` (LiDAR only)

```
lidar_0, lidar_1, ..., lidar_59,  speed,  steering_angle
```

| Nhóm | Cột | Đơn vị | Ghi chú |
|------|-----|--------|---------|
| LiDAR | `lidar_0` … `lidar_N-1` | mét | Crop [-60°,+60°], nội suy về N beams, clamp [0, max_range] |
| Label | `speed` | m/s | Tốc độ chuyên gia |
| Label | `steering_angle` | radian | Góc lái chuyên gia (âm = trái, dương = phải) |

> **N** = `target_beams` lúc thu thập. Mặc định **60**. File `dagger_dataset_sim_90.csv` dùng **90** beams.

---

### CSV từ `data_collection_odom_sim.py` (Odom only)

```
x, y, z, yaw, qx, qy, qz, qw, linear_vx, linear_vy, angular_wz, speed, steering_angle
```

| Cột | Đơn vị | Ghi chú |
|-----|--------|---------|
| `x`, `y`, `z` | mét | Vị trí trong world frame |
| `yaw` | radian | Góc quay quanh trục Z, tính từ quaternion |
| `qx`, `qy`, `qz`, `qw` | — | Quaternion orientation (dư thừa với yaw, có thể bỏ khi train) |
| `linear_vx`, `linear_vy` | m/s | Vận tốc tuyến tính |
| `angular_wz` | rad/s | Vận tốc góc |
| `speed` | m/s | **Label** — tốc độ chuyên gia |
| `steering_angle` | radian | **Label** — góc lái chuyên gia |

---

### CSV từ `data_collection_combined_sim.py` (LiDAR + Odom)

```
lidar_0 … lidar_59,  x, y, z, yaw, qx, qy, qz, qw,  linear_vx, linear_vy, angular_wz,  speed, steering_angle
   (60 cột)                    (11 cột)                                                    (2 cột)
```

Tổng: **73 cột** (với 60 beams), hoặc **83 cột** (với 90 beams).

---

## 4. File Huấn luyện

### 4.1 `train.py` — LiDAR Model

Script huấn luyện **Imitation Learning** (DAgger) cho model LiDAR. Mạng **MLP** đơn giản, output `[speed, steering_angle]`.

**Kiến trúc `DAggerMLP`:**
```
Input(N) → Linear(128) → ReLU → Dropout → Linear(64) → ReLU → Dropout → Linear(32) → ReLU → Linear(2)
```
Trong đó `N` = `input_dim` (số beams LiDAR).

**Cách chạy cơ bản:**

```bash
# Trường hợp 1: Dataset 60 beams (mặc định)
python3 train.py \
  --csv dagger_dataset_sim_4.csv \
  --model my_model_60.pth \
  --input_dim 60

# Trường hợp 2: Dataset 90 beams
python3 train.py \
  --csv dagger_dataset_sim_90.csv \
  --model model_sim_90.pth \
  --input_dim 90

# Trường hợp 3: Dataset Combined (60 LiDAR + 11 odom = 71 features thực sự đưa vào model)
# Lưu ý: train.py đọc input_dim cột đầu làm features, 2 cột cuối làm labels
# => với combined CSV 73 cột: input_dim = 71 (60 lidar + 11 odom)
python3 train.py \
  --csv combined_dataset_sim.csv \
  --model my_combined_model.pth \
  --input_dim 71
```

**Toàn bộ tham số:**

```bash
python3 train.py \
  --csv        dagger_dataset_sim_90.csv  # File CSV input
  --model      model_sim_90.pth          # File .pth output
  --input_dim  90                         # ← QUAN TRỌNG, xem mục 4.2
  --epochs     100                        # Số epoch tối đa
  --batch_size 32                         # Batch size
  --lr         0.001                      # Learning rate ban đầu
  --weight_decay 1e-5                     # L2 regularization
  --patience   15                         # Early stopping (epochs không cải thiện)
  --dropout    0.1                        # Dropout rate
```

**Đặc điểm nổi bật:**

- **Block-split** train/val theo thời gian (không random từng dòng) → giảm data leakage giữa các frame LiDAR liên tiếp giống nhau
- **Early stopping** theo val loss
- **LR Scheduler** `ReduceLROnPlateau` — tự giảm LR khi val loss không cải thiện
- Input LiDAR chuẩn hóa về [0, 1] bằng cách chia `LIDAR_MAX_RANGE = 10.0`
- Target (`speed`, `steering_angle`) chuẩn hóa **z-score** → tránh `speed` áp đảo `steering_angle` trong MSE
- Log **MAE riêng** cho từng output mỗi epoch

---

### 4.2 Tham số `--input_dim`: Cốt lõi cần hiểu

> [!CAUTION]
> **`--input_dim` PHẢI khớp chính xác với số cột LiDAR trong CSV.**
> Nếu truyền sai, `train.py` sẽ đọc cột `speed` và `steering_angle` nhầm sang cột LiDAR → **model học sai hoàn toàn**, không có lỗi báo trước khi train.

**Cách hoạt động bên trong `train.py`:**

```python
# train.py đọc CSV như sau:
self.inputs  = raw[:, :input_dim]           # input_dim cột ĐẦU → features
self.targets = raw[:, input_dim:input_dim+2] # 2 cột TIẾP THEO   → labels
```

Vậy với file CSV **73 cột** (60 LiDAR + 11 odom + 2 labels):

| `--input_dim` | Cột đọc làm features | Cột đọc làm labels | Kết quả |
|---------------|----------------------|--------------------|---------|
| `60` | `lidar_0…59` ✅ | `x, y` ❌ (sai!) | Model học sai |
| `71` | `lidar_0…59` + `x,y,z,yaw,qx,qy,qz,qw,vx,vy,wz` ✅ | `speed, steering_angle` ✅ | **Đúng** |
| `90` | đọc quá 73 cột → lỗi | — | Lỗi |

> [!TIP]
> `train.py` có **kiểm tra tự động**: nếu `input_dim` không khớp số cột LiDAR thực tế trong header, sẽ raise `ValueError` với thông báo rõ ràng trước khi bắt đầu train.

**Bảng tham chiếu nhanh:**

| File CSV | Số cột LiDAR | `--input_dim` đúng |
|----------|-------------|-------------------|
| `dagger_dataset_sim_4.csv` | 60 | `60` |
| `dagger_dataset_sim_90.csv` | 90 | `90` |
| `combined_dataset_sim.csv` (60 beam) | 60 lidar + 11 odom = 71 feature | `71` |
| `combined_dataset_sim.csv` (90 beam) | 90 lidar + 11 odom = 101 feature | `101` |

> Cách kiểm tra nhanh số cột trong CSV:
> ```bash
> head -1 dagger_dataset_sim_90.csv | tr ',' '\n' | wc -l
> # Trừ 2 (speed + steering_angle) = input_dim cần truyền
> ```

---

### 4.3 `train_odom.py` — Odometry Model

Script huấn luyện model **Trajectory Replay** dựa trên Odometry. Mục tiêu là **overfit chính xác** 1 quỹ đạo đã ghi, **không phải tổng quát hóa**.

**Input model (6 chiều cố định, không dùng `--input_dim`):**
```
[x, y, yaw, linear_vx, linear_vy, angular_wz]
```
> Bỏ `z`, `qx`, `qy`, `qz`, `qw` vì dư thừa với `yaw` đã có.

**Kiến trúc `OdomReplayMLP`:**
```
Input(6) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(2)
```
Không có Dropout mặc định (vì muốn overfit).

**Cách chạy:**
```bash
python3 train_odom.py \
  --csv      odom_dataset_sim.csv \
  --model    odom_replay_model.pth \
  --epochs   300          # Cao hơn bản LiDAR vì cần overfit kỹ
  --batch_size 64 \
  --lr       0.001 \
  --patience 30 \
  --dropout  0.0          # Giữ = 0 để overfit
  --weight_decay 0.0      # Giữ = 0 để overfit
  --hidden   64           # Số neuron mỗi hidden layer
```

**Điểm khác biệt quan trọng so với `train.py`:**

| Đặc điểm | `train.py` (LiDAR) | `train_odom.py` (Odom) |
|-----------|-------------------|----------------------|
| Mục tiêu | Tổng quát hóa — tránh va chạm | Overfit chính xác 1 quỹ đạo |
| Dropout | 0.1 (mặc định) | 0.0 |
| Weight decay | 1e-5 | 0.0 |
| Epochs | 100 | 300 |
| Patience | 15 | 30 |
| val_split | 0.2 | 0.1 (nhỏ hơn vì muốn nhiều data train) |
| Chuẩn hóa input | Chia `max_range=10.0` (LiDAR range) | z-score (x,y không có range cố định) |
| Lưu norm | `lidar_max_range` + `target_mean/std` | `input_mean/std` + `target_mean/std` |

---

## 5. File Inference (Chạy AI trên Sim)

### `ai_inference_sim.py` — LiDAR Inference

**Subscribe:** `/scan`  
**Publish:** `/drive`

```bash
ros2 run pure_pursuit_controller ai_inference_sim.py \
  --ros-args \
  -p model_path:=/sim_ws/.../model_sim_90.onnx \
  -p target_beams:=90 \
  -p ai_speed:=3.0 \
  -p max_range:=10.0
```

> [!IMPORTANT]
> `target_beams` trong inference **PHẢI bằng** `target_beams` lúc thu thập dữ liệu, và bằng `input_dim` lúc train. Nếu không khớp, model nhận input sai kích thước và crash hoặc cho kết quả ngẫu nhiên.

**Cơ chế nhận diện model cũ/mới:**

| `input_name` của ONNX | Loại model | Hành động |
|-----------------------|-----------|-----------|
| `'lidar_raw'` | Export bằng `DeployWrapper` | ONNX đã bake normalize/denormalize bên trong → không cần xử lý thêm |
| Khác (`'input'`, v.v.) | Export thô từ `torch.onnx.export` | Tự chia `/10.0` cho input + đọc `_norm.json` để denormalize output |

---

### `ai_inference_sim_odom.py` — Odometry Inference

**Subscribe:** `/ego_racecar/odom`  
**Publish:** `/drive`

```bash
ros2 run pure_pursuit_controller ai_inference_sim_odom.py \
  --ros-args \
  -p model_path:=/sim_ws/.../odom_replay_model.onnx \
  -p ai_speed:=3.0
```

**Thứ tự đặc trưng đầu vào phải khớp với lúc train:**
```python
features = [x, y, yaw, linear_vx, linear_vy, angular_wz]  # 6 chiều, đúng thứ tự
```

| `input_name` | Loại model | Hành động |
|-------------|-----------|-----------|
| `'odom_state'` | Export bằng DeployWrapper | ONNX xử lý normalize bên trong |
| Khác | Export thô | Đọc `_norm.json` để normalize input (`input_mean/std`) và denormalize output (`target_mean/std`) |

---

## 6. Normalization & File `_norm.json`

Sau mỗi lần train, một file `<model_name>_norm.json` được tự động tạo ra cùng thư mục với model `.pth`.

**Với `train.py` (LiDAR):**
```json
{
  "lidar_max_range": 10.0,
  "target_mean": [2.5, 0.01],
  "target_std":  [0.8, 0.15]
}
```

**Với `train_odom.py` (Odom):**
```json
{
  "input_cols":  ["x", "y", "yaw", "linear_vx", "linear_vy", "angular_wz"],
  "target_cols": ["speed", "steering_angle"],
  "input_mean":  [3.2, -1.1, 0.05, 2.3, 0.01, 0.02],
  "input_std":   [5.0, 4.0, 1.2, 0.7, 0.1, 0.3],
  "target_mean": [2.5, 0.01],
  "target_std":  [0.8, 0.15]
}
```

> [!NOTE]
> File `_norm.json` **PHẢI đi kèm với file `.onnx`** khi deploy. Nếu thiếu, inference node vẫn chạy nhưng output sẽ ở không gian chuẩn hóa (z-score) thay vì m/s và radian thực tế → xe chạy sai tốc độ/góc lái.

---

## 7. Workflow từ đầu đến cuối

### Workflow A: LiDAR-only (phổ biến nhất)

```bash
# --- Bước 1: Thu thập dữ liệu (Docker, song song với RRT expert) ---
ros2 run pure_pursuit_controller data_collection_sim.py \
  --ros-args -p target_beams:=60 -p dataset_path:=.../my_lidar_60.csv

# --- Bước 2: Train (local / Colab) ---
python3 train.py \
  --csv my_lidar_60.csv \
  --model my_model_60.pth \
  --input_dim 60           # ← PHẢI khớp target_beams ở bước 1

# --- Bước 3: Export sang ONNX ---
# (Dùng script export riêng hoặc torch.onnx.export trực tiếp)

# --- Bước 4: Inference (Docker) ---
ros2 run pure_pursuit_controller ai_inference_sim.py \
  --ros-args \
  -p model_path:=.../my_model_60.onnx \
  -p target_beams:=60      # ← PHẢI khớp target_beams ở bước 1
```

### Workflow B: LiDAR + Odom kết hợp

```bash
# --- Bước 1: Thu thập ---
ros2 run pure_pursuit_controller data_collection_combined_sim.py \
  --ros-args -p target_beams:=60   # → CSV 73 cột

# --- Bước 2: Kiểm tra số cột ---
head -1 combined_dataset_sim.csv | tr ',' '\n' | wc -l
# Kết quả: 73  →  input_dim = 73 - 2 = 71

# --- Bước 3: Train ---
python3 train.py \
  --csv combined_dataset_sim.csv \
  --model my_combined_model.pth \
  --input_dim 71           # ← 60 lidar + 11 odom

# --- Bước 4: Inference ---
# Cần viết node inference riêng đọc cả /scan và /ego_racecar/odom,
# ghép thành vector 71 chiều rồi feed vào model.
```

---

## 8. Bảng tra cứu nhanh: Phối hợp Data — Train — Inference

| File CSV | Số cột | `--input_dim` | `target_beams` inference | Script Inference |
|----------|--------|---------------|--------------------------|-----------------|
| `dagger_dataset_sim_4.csv` | 62 | `60` | `60` | `ai_inference_sim.py` |
| `dagger_dataset_sim_90.csv` | 92 | `90` | `90` | `ai_inference_sim.py` |
| `odom_dataset_sim.csv` | 13 | N/A (fixed 6) | N/A | `ai_inference_sim_odom.py` |
| `combined_dataset_sim.csv` (60 beam) | 73 | `71` | N/A (cần node custom) | custom |

> [!TIP]
> **Quy tắc nhớ nhanh:**
> - `--input_dim` = tổng số cột CSV − 2 (luôn bỏ 2 cột label cuối cùng)
> - `target_beams` trong inference = `target_beams` lúc thu thập = `--input_dim` lúc train (với model LiDAR-only)
> - Chỉ có model LiDAR-only mới cần khớp `target_beams` với inference. Model Odom không cần.
