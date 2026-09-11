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

### 1.4 Giới Hạn Hành Động Bất Đối Xứng & Tránh Bẫy Gradient Bằng 0 (Zero-Gradient Bottleneck)
Offset hành động $a_R(s) = [\Delta v, \Delta \delta]$ được ánh xạ bất đối xứng:
- $\Delta v \in [-1.5, +2.5]\text{ m/s}$: Cho phép giảm tốc sâu khi cua/gặp chướng ngại vật và tăng tốc mạnh ở đường thẳng.
- $\Delta \delta \in [-0.035, +0.035]\text{ rad} \approx \pm 2.0^\circ$: Giữ steering bẻ mượt.

> [!CAUTION]
> **Zero-Gradient Bottleneck Trap:** 
> Khi khởi tạo zero-init ($\mathbf{W}_{\text{out}} = \mathbf{0}$), nếu dùng `torch.relu(u_v)` tại điểm $u_v = 0$, đạo hàm của `relu(0)` bằng $0$. Điều này dẫn đến gradient của layer đầu ra tốc độ bị triệt tiêu hoàn toàn ($\nabla_{\mathbf{W}_v} \mathcal{L} = 0$), khiến trọng số tốc độ không bao giờ học được!
> **Giải pháp:** Sử dụng `torch.where(u_v >= 0, torch.tanh(u_v) * 2.5, torch.tanh(u_v) * 1.5)` để đảm bảo gradient tại điểm khởi tạo $u_v = 0$ luôn dương (lần lượt là $2.5$ và $1.5$).

---

### 1.5 Quy Chuẩn Dữ Liệu: PyTorch (`.pth`) vs ONNX (`.onnx`)
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

---

## 5. Bảng Các Tham Số Cần Tuning Theo Ý Muốn

### 5.1 Tham Số Trong `residual_env_wrapper.py` (Hàm Reward)

| Tham số | Dòng | Giá trị mặc định | Tác dụng | Gợi ý điều chỉnh |
|:--|:--|:--|:--|:--|
| `front_dist_threshold` | L219 | `3.5m` | Ngưỡng phân biệt đường thẳng / cua | Giảm xuống `3.0m` nếu xe phanh sớm quá |
| `speed_reward_scale` | L220 | `2.0` | Mức thưởng khi xe chạy nhanh trên đường thẳng | Tăng lên `3.0` nếu xe vẫn không tăng tốc |
| `v_target_corner` scale | L223 | `front_dist * 1.0` | Tốc độ mục tiêu khi vào cua | Giảm `1.0 → 0.8` để xe phanh sâu hơn trước cua |
| `v_target_corner` min | L223 | `max(1.5, ...)` | Tốc độ tối thiểu ở góc cua hẹp nhất | Giảm xuống `1.0` nếu cua rất gấp |
| `corner_speed_penalty_scale` | L224 | `2.0` | Cường độ phạt khi sai tốc độ trong cua | Tăng lên `3.0` để phanh cua dứt khoát hơn |
| `steer_bias_penalty_scale` | L221 | `2.0` (L2) | Ép góc bẻ dư về 0 trên đường thẳng | Tăng lên `3.0` nếu xe vẫn bẻ lệch góc |
| `jerk_penalty` | L228 | `0.2` | Phạt thay đổi action đột ngột giữa 2 bước | Tăng lên `0.5` nếu xe rung vô lăng |
| `stopping_penalty` scale | L232 | `3.0` | Phạt mạnh khi xe đứng yên / bò chậm | Tăng lên `5.0` nếu xe hay dừng giữa đường |

### 5.2 Tham Số Trong `rrl_model.py` (Action Bounds)

| Tham số | Dòng | Giá trị mặc định | Tác dụng | Gợi ý điều chỉnh |
|:--|:--|:--|:--|:--|
| `v_boost` | L61 | `2.5 m/s` | Biên độ tăng tốc tối đa của RRL | Giảm xuống `1.5` nếu CBF can thiệp liên tục |
| `v_brake` | L62 | `1.5 m/s` | Biên độ phanh tối đa của RRL | Tăng lên `2.0` để phanh sâu hơn trước cua gấp |
| `steer_max` | L63 | `0.035 rad ≈ 2°` | Biên độ bẻ lái tối đa của RRL | Tăng lên `0.05 rad ≈ 2.87°` nếu cần điều chỉnh lái nhiều hơn |

### 5.3 Tham Số Trong `train_ppo_rrl.py` (Hyper-parameter PPO)

| Tham số | Flag / Dòng | Giá trị mặc định | Tác dụng | Gợi ý điều chỉnh |
|:--|:--|:--|:--|:--|
| `learning_rate` | `--learning_rate` | `3e-4` | Tốc độ học của Adam optimizer | Giảm xuống `1e-4` nếu policy collapse sớm |
| `total_timesteps` | `--total_timesteps` | `200000` | Tổng số bước train | Tăng lên `500000` nếu muốn policy mượt và chín hơn |
| `rollout_horizon` | `--rollout_horizon` | `2048` | Độ dài rollout mỗi epoch | Tăng lên `4096` để GAE có nhiều dữ liệu hơn |
| `batch_size` | `--batch_size` | `128` | Kích thước minibatch PPO | Giảm xuống `64` nếu muốn gradient update ổn định hơn |
| `gamma` | `--gamma` | `0.998` | Hệ số chiết khấu tương lai | Giữ nguyên; giảm xuống `0.99` nếu policy ưu tiên thưởng ngắn hạn |
| `entropy_coeff` | L152 | `0.05` | Khuyến khích exploration; ngăn policy collapse | Range `[0.03, 0.1]`; giảm xuống `0.03` khi gần converge |
| `clip_eps` | `--clip_eps` | `0.2` | PPO clip ratio | Giữ nguyên; giảm xuống `0.1` nếu policy update quá nhảy |

### 5.4 Tham Số Trong `rrl_safe_controller_node.py` (Deployment)

| Tham số | Dòng | Giá trị mặc định | Tác dụng | Gợi ý điều chỉnh |
|:--|:--|:--|:--|:--|
| EMA alpha `0.35` | L289 | `0.35` | Bộ lọc mượt action RRL (0=không lọc, 1=lọc hoàn toàn) | Tăng lên `0.5` nếu xe rung; giảm xuống `0.2` nếu cần phản ứng nhanh |
| `v_clip` min | L294 | `0.5 m/s` | Tốc độ tối thiểu sau khi cộng RRL | Giảm xuống `0.0` để cho phép dừng hoàn toàn |
| `v_clip` max | L294 | `7.0 m/s` | Tốc độ tối đa sau khi cộng RRL | Giảm xuống `5.0` nếu muốn giới hạn speed an toàn hơn |
| CBF `d_min` | L80 | `0.30m` | Khoảng cách an toàn tối thiểu CBF | Giảm xuống `0.20m` nếu CBF can thiệp quá sớm |
| CBF `gamma` | L80 | `1.5` | Hệ số phản ứng CBF | Tăng lên `2.5` để CBF phản ứng mạnh hơn |

---

## 6. Nguyên Nhân Học Thiên Hướng & Các Bug Đã Sửa

### Bug 1 (Đã sửa): `a_R² magnitude penalty` làm Δv bị ép về 0
- **Triệu chứng:** `RRL: [Δv=+0.00, ...]` không bao giờ tăng tốc
- **Nguyên nhân:** `0.05 * np.sum(a_R ** 2)` phạt cả Δv dương đúng đắn → PPO chọn Δv=0 an toàn hơn
- **Khắc phục:** Chỉ giữ penalty `jerk (a_R - prev_a_R)`, loại bỏ hạng magnitude

### Bug 2 (Đã sửa): L1 steer penalty `|Δδ|` tạo điểm Nash lệch một phía
- **Triệu chứng:** `RRL: [... Δsteer= -0.6°]` cố định không đổi
- **Nguyên nhân:** Gradient của `|Δδ|` là hằng số $\pm 1$ → tạo điểm tối ưu lệch khỏi 0 khi track không cân xứng (nhiều cua trái)
- **Khắc phục:** Đổi sang `L2 penalty = 2.0 * Δδ²` → gradient tỷ lệ với |Δδ|, kéo về 0 đối xứng

### Bug 3 (Đã sửa): `front_dist` tính từ obs cũ (trước action) thay vì obs mới
- **Triệu chứng:** Reward signal chậm ~1 step → PPO nhận tín hiệu không đúng thực tế
- **Khắc phục:** Dùng `obs` (sau bước) thay vì `gym_raw_obs` (trước bước)

### Bug 4 (Đã sửa): Entropy coefficient `0.01` quá nhỏ → Policy collapse sớm
- **Triệu chứng:** Sau ~50k steps, policy bị khóa cứng vào một behavior
- **Khắc phục:** Tăng `entropy_coeff` từ `0.01` lên `0.05`

### Bug 5 (Đã sửa): Clip range trong `get_action` không khớp `_map_action`
- **Triệu chứng:** log_prob bị bias → PPO cập nhật gradient sai
- **Nguyên nhân:** `v_clamped = clamp(raw, -0.3, 2.5)` nhưng `_map_action` hỗ trợ braking tới `-1.5`
- **Khắc phục:** Đổi thành `clamp(raw, -1.5, 2.5)` để phù hợp
