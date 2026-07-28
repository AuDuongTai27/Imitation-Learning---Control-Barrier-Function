# F1TENTH Imitation Learning — Tài liệu kỹ thuật

Dự án gồm **2 pipeline huấn luyện độc lập** cho xe tự lái F1TENTH bằng Imitation Learning (DAgger), dùng chuyên gia Pure Pursuit làm nguồn dữ liệu điều khiển:

| Pipeline | File train | Input | Mục tiêu |
|---|---|---|---|
| **LiDAR-based** | `train.py` | Khoảng cách LiDAR (60/90 tia) | Lái tự động tổng quát, cảm nhận môi trường |
| **Odometry-based** | `train_odom.py` | Vị trí + vận tốc tuyệt đối (x, y, yaw, v) | Replay lại đúng 1 quỹ đạo đã ghi |

Đây là 2 cách tiếp cận **khác bản chất hoàn toàn**, không thay thế nhau — chi tiết bên dưới.

---

## 1. Pipeline LiDAR (`train.py`)

### 1.1. Input

| Thuộc tính | Giá trị |
|---|---|
| Nguồn dữ liệu | `/scan` (LaserScan), crop về góc quét cố định (±60° hoặc ±90° tùy phiên bản data collection) |
| Số chiều | 60 hoặc 90 (khớp đúng số beam đã thu thập — bắt buộc set đúng `--input_dim`) |
| Đơn vị gốc | Mét, giá trị trong khoảng `[0, max_range]` (mặc định `max_range=10.0`) |
| Tiền xử lý | Downsample bằng nội suy tuyến tính (`np.interp`), NaN/Inf được thay bằng `max_range` |
| Chuẩn hóa lúc train | Chia cho `LIDAR_MAX_RANGE` → đưa về `[0, 1]` |

Input là **cảm nhận tương đối quanh xe** (khoảng cách tới vật cản theo từng góc) — đây là điểm mấu chốt giúp model tổng quát hóa: dù xe ở bất kỳ vị trí nào trên track, hình dạng LiDAR nhìn thấy tương tự nhau ở các tình huống tương tự (vd: "tường bên trái gần" luôn có pattern gần giống nhau bất kể tọa độ tuyệt đối).

### 1.2. Output

| Output | Đơn vị | Range thực tế |
|---|---|---|
| `linear_v` (speed) | m/s | 0 → `ai_speed` (giới hạn an toàn, mặc định 1.0–3.0) |
| `angular_z` (steering_angle) | rad | −0.41 → 0.41 (giới hạn cơ khí góc lái) |

Output được **chuẩn hóa z-score lúc train** (`(x - mean) / std`), lưu lại `target_mean`/`target_std` trong file `*_norm.json`. Khi export ONNX (`export_onnx.py`), bước denormalize được **bake thẳng vào đồ thị ONNX** — nên model triển khai thực tế nhận LiDAR thô (mét) và trả thẳng `(speed, steering)` thật, không cần code inference tự tính lại.

### 1.3. Thuật toán

- **Kiến trúc**: MLP 4 lớp `Linear(input_dim→128) → ReLU → Dropout → Linear(128→64) → ReLU → Dropout → Linear(64→32) → ReLU → Linear(32→2)`
- **Loss**: MSE (Mean Squared Error) giữa `(speed, steering)` dự đoán và của chuyên gia Pure Pursuit
- **Optimizer**: Adam, `lr=0.001`, `weight_decay=1e-5` (regularization nhẹ)
- **LR Scheduler**: `ReduceLROnPlateau` — giảm nửa learning rate nếu val loss không cải thiện sau 5 epoch
- **Train/Val split**: chia theo **block liên tục theo thời gian** (hoặc theo `episode` nếu CSV có cột đó) — không random theo dòng, để tránh data leakage giữa các frame LiDAR liền kề gần giống hệt nhau
- **Early stopping**: dừng nếu val loss không cải thiện sau `patience` epoch (mặc định 15)
- **Regularization**: Dropout (mặc định 0.1) + weight decay — **mục tiêu là tổng quát hóa**, khác hẳn pipeline Odom

### 1.4. Cơ chế hoạt động (DAgger)

Khác Behavior Cloning thuần túy (chỉ học từ dữ liệu chuyên gia thu 1 lần), DAgger là quy trình lặp:
1. Thu dữ liệu ban đầu từ chuyên gia Pure Pursuit.
2. Train model, cho model tự lái (closed-loop).
3. Ở những chỗ model sai (vd 1 khúc cua cụ thể model luôn đi thẳng), thu thêm dữ liệu đúng tại chính điểm lỗi đó.
4. Gộp vào dataset, train lại.

Đây là lý do khi phát hiện model sai ở 1 khúc cua cố định, cách sửa đúng không phải là chỉnh loss function trước, mà là **quay lại đúng khúc cua đó thu thêm data** — sửa đúng vào phân bố dữ liệu đang thiếu.

### 1.5. Tính an toàn

- **Có khả năng tổng quát hóa** sang các vị trí/tình huống chưa từng thấy hệt, miễn pattern LiDAR tương tự đã có trong training data.
- **Rủi ro nội tại của Imitation Learning**: covariate shift / compounding error — sai số nhỏ tích lũy dần vì input(t+1) phụ thuộc output(t) do chính model tạo ra, không phải quỹ đạo chuyên gia gốc.
- **Không có safety layer riêng** trong model — toàn bộ an toàn phụ thuộc vào `np.clip()` giới hạn `speed`/`steering` ở tầng inference. Không có cơ chế phát hiện "model không chắc chắn" (uncertainty estimation) hay fallback tự động dừng xe khi input bất thường.
- **Sim-to-real gap**: model train hoàn toàn từ LiDAR sim, chưa từng thấy noise/đặc tính LiDAR thật — khuyến nghị luôn test tốc độ thấp (`ai_speed` nhỏ) lần đầu trên xe thật.

### 1.6. Độ trễ (Latency)

- Model là MLP nhỏ (~4 lớp, vài chục nghìn tham số) — **không có state/memory** (thuần phản ứng theo input hiện tại, không phải RNN/Transformer) → độ trễ suy luận cực thấp, dưới 1ms trên CPU Jetson thông thường, không đáng kể so với tần số `/scan` (thường 10–40Hz).
- Độ trễ hệ thống thực tế chủ yếu đến từ: driver LiDAR → ROS 2 message passing → tiền xử lý (`preprocess_scan`, dùng `np.interp`) → ONNX Runtime session → publish `/drive` → VESC. Không có bottleneck từ bản thân model.
- Trên Jetson, `onnxruntime` đôi khi gặp lỗi cấp phát CPU (`std::bad_alloc`) → khuyến nghị dùng PyTorch trực tiếp (`ai_inference_real_pytorch.py`, JIT-compiled bằng `torch.jit.script`) nếu gặp vấn đề ổn định.

---

## 2. Pipeline Odometry (`train_odom.py`)

### 2.1. Input

| Thuộc tính | Giá trị |
|---|---|
| Nguồn dữ liệu | `/ego_racecar/odom` (Odometry), đồng bộ với `/drive` |
| Số chiều | 6: `[x, y, yaw, linear_vx, linear_vy, angular_wz]` |
| Đơn vị gốc | Mét (x, y), radian (yaw), m/s (vx, vy), rad/s (wz) |
| Cột bị loại bỏ | `z`, quaternion `(qx, qy, qz, qw)` — dư thừa vì `z≈0` (xe phẳng) và `yaw` đã được tính sẵn từ quaternion |
| Chuẩn hóa lúc train | z-score: `(x - input_mean) / input_std` (không dùng hằng số cố định như LiDAR, vì `x, y` không có range biết trước) |

**Khác biệt cốt lõi so với LiDAR**: input là **tọa độ/vận tốc tuyệt đối trong world frame**, không phải cảm nhận môi trường tương đối. Model không hề "nhìn thấy" vật cản hay hình dạng track.

### 2.2. Output

| Output | Đơn vị | Range thực tế |
|---|---|---|
| `speed` | m/s | 0 → `ai_speed` |
| `steering_angle` | rad | −0.41 → 0.41 |

Cũng chuẩn hóa z-score, denormalize được bake vào ONNX qua `export_onnx_odom.py` (tương tự pipeline LiDAR).

### 2.3. Thuật toán

- **Kiến trúc**: MLP nhỏ hơn hẳn — `Linear(6→64) → ReLU → Linear(64→64) → ReLU → Linear(64→2)`, **không Dropout** (mặc định `dropout=0`)
- **Loss**: MSE
- **Optimizer**: Adam, `weight_decay=0.0` mặc định (tắt hẳn regularization)
- **Epochs**: cao hơn hẳn (300 vs 100), `patience=30`
- **Train/val split**: block theo thời gian (không có khái niệm episode nhiều track khác nhau)

### 2.4. Cơ chế hoạt động — Trajectory Replay, KHÔNG phải điều khiển tổng quát

Đây là điểm quan trọng nhất cần hiểu: vì input là tọa độ tuyệt đối, quá trình train về bản chất là **ép model học thuộc lòng 1 bảng tra cứu** "ở tọa độ/hướng/vận tốc này thì lái góc này" — không có khái niệm tổng quát hóa sang tình huống mới. Đây là lý do:
- `dropout=0`, `weight_decay=0` **mặc định** — vì ở đây **overfit chính là mục tiêu**, ngược hoàn toàn tinh thần regularization của pipeline LiDAR.
- `epochs` và `patience` cao hơn — cần train kỹ để model khớp sát nhất có thể với quỹ đạo gốc.

### 2.5. Tính an toàn — **Hạn chế nghiêm trọng, cần lưu ý kỹ**

- **Không tổng quát hóa được sang track khác** hoặc vị trí khởi động khác.
- **Không có nhận thức về vật cản/môi trường** — nếu có vật cản bất ngờ xuất hiện không nằm trong dữ liệu train, model sẽ không phản ứng gì (vì input không hề chứa thông tin đó).
- **Nhạy với sai số tích lũy (drift) của odometry**: xe thật không có ground-truth position chính xác như sim — encoder/IMU drift theo thời gian khiến `(x, y, yaw)` ước lượng ngày càng lệch so với thực tế, đẩy input ra ngoài phân bố train.
- **Bắt buộc khởi động đúng vị trí/hướng đã ghi lúc thu thập** — sai lệch điểm bắt đầu dù nhỏ sẽ khiến model dự đoán sai ngay từ giây đầu tiên.
- **Khuyến nghị**: chỉ dùng cho mục đích thử nghiệm/demo replay trên chính track mô phỏng đã thu thập, **không dùng làm bộ điều khiển chính cho xe thật** trừ khi có thêm cơ chế bù trừ vị trí (localization) đáng tin cậy.

### 2.6. Độ trễ (Latency)

- Model còn nhỏ hơn pipeline LiDAR (input 6 chiều, 2 hidden layer 64 neuron) → độ trễ suy luận không đáng kể, thấp hơn cả pipeline LiDAR.
- Tần số `/ego_racecar/odom` trong sim thường cao hơn `/scan` (tùy cấu hình plugin odometry), nên tần số điều khiển thực tế phụ thuộc topic này thay vì tốc độ model.
- Không có xử lý ảnh/tín hiệu nặng (không cần `np.interp` như LiDAR) → tiền xử lý gần như tức thời, chỉ có phép tính quaternion → yaw (`atan2`, chi phí không đáng kể).

---

## 3. So sánh nhanh 2 pipeline

| Tiêu chí | LiDAR (`train.py`) | Odometry (`train_odom.py`) |
|---|---|---|
| Input | Cảm nhận môi trường (tương đối) | Tọa độ/vận tốc tuyệt đối |
| Tổng quát hóa | Có | Không (chỉ replay đúng 1 quỹ đạo) |
| Regularization | Dropout + weight decay | Tắt (mục tiêu overfit) |
| Dùng được cho xe thật độc lập | Có (với lưu ý sim-to-real gap) | Không khuyến nghị (rủi ro drift + không nhận thức môi trường) |
| Độ phức tạp model | Trung bình (input 60–90 chiều) | Thấp (input 6 chiều) |
| Rủi ro chính | Compounding error, thiếu data ở khúc cua hiếm gặp | Drift odometry, sai vị trí khởi động |

## 4. File liên quan

```
train.py                     # Train pipeline LiDAR
export_onnx.py                # Export .pth -> .onnx (LiDAR), bake normalize/denormalize
ai_inference_sim.py           # Inference ONNX trong sim (LiDAR)
ai_inference_real_pytorch.py  # Inference PyTorch trực tiếp trên Jetson (LiDAR)
check_steering_distribution.py# Kiểm tra mất cân bằng phân bố góc lái trong dataset

train_odom.py                 # Train pipeline Odometry
export_onnx_odom.py            # Export .pth -> .onnx (Odom), bake normalize/denormalize
ai_inference_sim_odom.py       # Inference ONNX trong sim (Odom)
```

Mỗi file `.pth` sau khi train đều đi kèm 1 file `*_norm.json` cùng tên — **bắt buộc giữ 2 file này đi cùng nhau**, vì file `.pth` không tự chứa thông tin chuẩn hóa.