# Hướng Dẫn Kỹ Thuật & Lý Thuyết: Residual Reinforcement Learning (RRL) + CBF Safety Filter Cho F1TENTH

Tài liệu này hướng dẫn chi tiết về **Lý thuyết, Cấu trúc Mã nguồn, Quy trình Train Local, Các Lệnh Triển Khai Kiểm Thử** và **Phương Pháp Sửa Lỗi RRL** kết hợp với mô hình **Deep Imitation Learning (DIL - DAgger)** và bộ lọc an toàn **Control Barrier Function (CBF-QP)**.

---

## 1. Lý Thuyết Residual Reinforcement Learning (RRL / RPL)

### 1.1 Tối Ưu Hóa Dựa Trên Baseline Đông Cứng (Frozen Baseline)
Trong bài toán đua xe tự hành F1TENTH, việc huấn luyện Reinforcement Learning (RL) từ đầu (from scratch) thường gặp phải các vấn đề:
- Tốn nhiều thời gian khám phá (exploration).
- Dễ gây va chạm đâm tường liên tục ở giai đoạn đầu khiến model chậm hội tụ.

Giải pháp **Residual Reinforcement Learning (RRL)** sử dụng một mô hình chuyên gia học trước (DIL Baseline Policy $\pi_{\text{DIL}}$ đã huấn luyện bằng DAgger). Mô hình DIL này được **đóng băng (frozen)** (`requires_grad = False`).

Chính sách tổng cộng gửi xuống bộ điều khiển xe là:
$$a_{\text{total}}(s) = a_{\text{DIL}}(s) + a_R(s)$$

Trong đó:
- $a_{\text{DIL}}(s) = [v_{\text{DIL}}, \delta_{\text{DIL}}]$ là vận tốc và góc lái do mô hình DIL tính toán.
- $a_R(s) = [\Delta v, \Delta \delta]$ là offset điều chỉnh do chính sách RRL ($\pi_R$) học bằng PPO.

---

### 1.2 Zero-Initialization Trick (Khởi Tạo Lớp Cuối Bằng 0)
Đây là kỹ thuật cốt lõi để đảm bảo quá trình PPO bootstrapping diễn ra mượt mà:
- Toàn bộ trọng số (weights) và độ lệch (bias) của layer tuyến tính cuối cùng trong Actor Network ($\pi_R$) được khởi tạo bằng **0**:
  $$\mathbf{W}_{\text{out}} = \mathbf{0}, \quad \mathbf{b}_{\text{out}} = \mathbf{0}$$
- **Ý nghĩa:** Tại thời điểm bắt đầu huấn luyện ($t=0$), ta có:
  $$a_R(s) \approx [0, 0] \implies a_{\text{total}}(s) \approx a_{\text{DIL}}(s)$$
- Điều này giúp xe di chuyển an toàn như mô hình DIL expert ban đầu, không bị mất lái đâm tường ngay ở bước rollout đầu tiên.

---

### 1.3 Giới Hạn Hành Động (Action Bounding & Tanh Scaling)
Offset hành động $a_R(s)$ được giới hạn trong một khoảng an toàn để tránh RRL lấn áp hoàn toàn DIL baseline:
$$a_R(s) = \text{Tanh}(\mu_{\text{net}}(s)) \odot \text{scale}$$

Véctơ scaling mặc định:
$$\text{scale} = [1.0, 0.05]$$
- $\Delta v \in [-1.0, +1.0]\text{ m/s}$: Tăng/giảm tốc độ tối đa $1.0\text{ m/s}$.
- $\Delta \delta \in [-0.05, +0.05]\text{ rad} \approx [-2.87^\circ, +2.87^\circ]$: Tinh chỉnh góc lái.

---

### 1.4 Quy Chuẩn Dữ Liệu: PyTorch (`.pth`) vs ONNX (`.onnx`)
> [!IMPORTANT]
> **Quy tắc vàng về Preprocessing & Denormalization:**

| Tiêu chí | Model PyTorch (`.pth`) | Model ONNX (`.onnx` - `final_combined_37500`) |
| :--- | :--- | :--- |
| **Dữ liệu Scan Input** | Scan normalized $[0.0, 1.0]$ (chia `max_range = 10.0`) | Scan **đơn vị mét thực tế $[0.0, 10.0]$** (tên layer `lidar_raw`) |
| **Đầu ra (Output)** | Raw normalized output | **GIÁ TRỊ VẬT LÝ THỰC TẾ** ($v\text{ m/s}, \delta\text{ rad}$) |
| **Giải chuẩn hóa (Denorm)** | Bắt buộc: `raw * target_std + target_mean` | **KHÔNG giải chuẩn hóa** (đã đóng gói trong đồ thị ONNX) |

---

## 2. Cấu Trúc Mã Nguồn Trong Repo

Các file RRL được đặt tại package `pure_pursuit_controller`:

```
src/pure_pursuit_controller/pure_pursuit_controller/
├── training/
│   └── rrl/
│       ├── residual_env_wrapper.py   # Gym Wrapper xử lý state, DIL baseline, CBF & reward
│       ├── rrl_model.py              # PyTorch Actor-Critic Network (Zero-Init + Tanh)
│       ├── train_ppo_rrl.py          # Script huấn luyện PPO Local (FPS > 300)
│       └── convert_rrl_to_onnx.py    # Script chuyển đổi checkpoint PyTorch -> ONNX
├── controllers/
│   └── rrl_safe_controller_node.py   # ROS 2 Node triển khai mô hình RRL + CBF
└── models/
    ├── final_combined_37500.onnx     # DIL Baseline Model (ONNX)
    ├── final_combined_37500_norm.json # Normalization parameters
    └── rrl_ppo_model.onnx            # Trained RRL Model (ONNX)
```

---

## 3. Tóm Tắt Các Lệnh Kiểm Thử & Triển Khai Quan Trọng

### 3.1 Huấn Luyện RRL Local (Tốc độ cao ~300 FPS)
```bash
python3 /home/adt/f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/training/rrl/train_ppo_rrl.py \
    --total_timesteps 100000 \
    --rollout_horizon 2048 \
    --batch_size 128 \
    --use_cbf
```

---

### 3.2 Lệnh Triển Khai Kiểm Thử Cách Ly Chẩn Đoán (Diagnostic Modes)

Trong file [rrl_safe_controller_node.py](file:///home/adt/f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/controllers/rrl_safe_controller_node.py), bạn có thể bật/tắt độc lập RRL và CBF bằng tham số `--ros-args`:

#### Mode 1: CHỈ MÌNH DIL (Pure DIL Mode)
Dùng để kiểm tra baseline DIL lái có chuẩn và mượt không (tắt hoàn toàn RRL và CBF):
```bash
ros2 run pure_pursuit_controller rrl_safe_controller_node.py --ros-args \
    -p dagger_model_path:=/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/models/final_combined_37500.onnx \
    -p enable_rrl:=false \
    -p use_cbf:=false
```

#### Mode 2: DIL + CBF (An Toàn Không Có RRL)
Dùng để kiểm tra xem bộ lọc CBF có gây ra hiện tượng khựng/giật vô lăng hay không:
```bash
ros2 run pure_pursuit_controller rrl_safe_controller_node.py --ros-args \
    -p dagger_model_path:=/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/models/final_combined_37500.onnx \
    -p enable_rrl:=false \
    -p use_cbf:=true
```

#### Mode 3: ĐẦY ĐỦ (DIL + RRL + CBF)
Chạy đầy đủ hệ thống RRL điều chỉnh DIL kết hợp CBF:
```bash
ros2 run pure_pursuit_controller rrl_safe_controller_node.py --ros-args \
    -p rrl_model_path:=/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/models/rrl_ppo_model.onnx \
    -p dagger_model_path:=/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/models/final_combined_37500.onnx \
    -p enable_rrl:=true \
    -p use_cbf:=true
```

---

## 4. Nguyên Nhân RRL Không Mượt & Phương Pháp Sửa Lỗi (Remedies)

Nếu Chế độ 1 (Pure DIL) chạy mượt nhưng khi bật RRL bị giật/lắc vô lăng:

### Nguyên nhân:
1. **Thiếu Action Smoothness / Rate Penalty trong Reward**: Trong môi trường RL cũ, hàm reward chỉ thưởng $v_x$ và phạt va chạm mà **không phạt sự thay đổi đột ngột giữa 2 bước hành động kế tiếp $(\Delta a_t - \Delta a_{t-1})^2$**. Điều này làm chính sách PPO liên tục nhảy offset cực đại để tối ưu reward tức thời.
2. **Action Scaling Bị Rộng**: Mức bù góc lái $\pm 0.05\text{ rad} \approx \pm 2.87^\circ$ nếu dao động ở tần số 30Hz sẽ làm tay lái rung lắc.

### Giải Pháp Sửa Lỗi Được Đề Xuất:

1. **Thêm Phạt Mượt Hành Động (Action Rate & Magnitude Penalty) vào `residual_env_wrapper.py`**:
   $$\text{Reward} = \tau_1 v_x + \tau_2 v_y^2 - \beta_1 \|\Delta a_t - \Delta a_{t-1}\|^2 - \beta_2 \|\Delta a_t\|^2$$
2. **Thu Nhỏ Biên Độ Action Scale (`rrl_model.py`)**:
   Giảm scale xuống $\text{scale} = [0.3\text{ m/s}, 0.02\text{ rad} \ (\approx 1.15^\circ)]$ để RRL chỉ tinh chỉnh nhẹ nhàng, không can thiệp thô bạo vào DIL.
3. **Thêm Bộ Lọc Mượt EMA (Exponential Moving Average) ở Node Deployment**:
   $$a_{R, \text{filtered}} = \alpha \cdot a_{R, \text{new}} + (1 - \alpha) \cdot a_{R, \text{prev}} \quad (\alpha \approx 0.25)$$
