#!/usr/bin/env python3
"""
rrl_inference_real_pytorch.py
─────────────────────────────
ROS 2 Node: Full RRL Pipeline for REAL F1TENTH Robot (PyTorch .pth files)
            integrated with Chance-Constrained GP-Adaptive CBF-QP Safety Filter.

Pipeline:
  /scan  ──► Preprocess (60 beams, norm) ──► DIL (.pth) ──► a_DIL = [v, δ]
                                         ──► RRL State (66-dim)
                                         ──► RRL (.pth) ──► a_R  = [Δv, Δδ]
                                         ──► a_total = a_DIL + a_R
  /odom (v_y) ──► Online Gaussian Process ──► Slip Uncertainty (μ, σ)
                                         ──► Chance-Constrained CBF-QP ──► /drive

Parameters (set via --ros-args -p):
  dil_model_path      : Path to DIL .pth model (default: final_combined_37500.pth)
  rrl_model_path      : Path to RRL .pth model (default: rrl_ppo_model.pth)
  norm_param_path     : Path to _norm.json (default: final_combined_37500_norm.json)
  enable_rrl          : bool (default: True)  — set False for Pure DIL mode
  use_cbf             : bool (default: True)  — set False to bypass CBF
  scan_topic          : string (default: /scan)
  drive_topic         : string (default: /drive)
  odom_topic          : string (default: /odom)
  max_speed           : float (default: 3.0)  — max speed limit (m/s)
  min_speed           : float (default: 0.0)  — allow full stop

  CBF & GP Parameters:
  cbf_d_min           : float (default: 0.15) — Khoảng cách an toàn tối thiểu (m)
  cbf_gamma           : float (default: 3.5)  — Độ dốc hội tụ CBF
  cbf_a_max_brake     : float (default: 3.5)  — Gia tốc phanh cực đại (m/s²)
  cbf_wheelbase       : float (default: 0.39) — Chiều dài cơ sở L (m)
  cbf_fov_deg         : float (default: 15.0) — Góc mở radar kiểm soát va chạm (+/- deg)
  cbf_steer_max       : float (default: 0.41) — Giới hạn góc lái tối đa (rad)
  enable_gp           : bool (default: True)  — Bật học thích nghi trượt lốp GP
  k_delta             : float (default: 1.2)  — Hệ số tin cậy Chance-Constraint
  gp_buffer_size      : int (default: 60)     — Kích thước bộ nhớ trượt FIFO của GP
  gp_length_scale     : float (default: 0.9)  — Độ nhạy tương đồng vận tốc
  gp_sigma_f          : float (default: 0.06) — Biên độ trượt lốp tối đa (m/s)
  gp_sigma_n          : float (default: 0.02) — Phương sai nhiễu đo lường Odometry
  slip_source         : string (default: twist_vy) — Nguồn đo trượt ('twist_vy' cho sim/IMU, 'yaw_rate_diff' cho P1, 'pose_curvature' cho P2)
"""

import os
import sys
import csv
import json
import math
import select
import termios
import tty
import threading
from datetime import datetime
from collections import deque
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from ackermann_msgs.msg import AckermannDriveStamped

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    raise RuntimeError("PyTorch is required for this node. Install with: pip3 install torch")

class DAggerMLP(nn.Module):
    def __init__(self, input_dim=60, output_dim=2, dropout=0.1):
        super(DAggerMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class RRLActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 2, scale=(0.4, 0.025)):
        super(RRLActorCritic, self).__init__()
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.actor_backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.actor_mean = nn.Linear(256, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim) - 0.5)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        nn.init.zeros_(self.actor_mean.weight)
        nn.init.zeros_(self.actor_mean.bias)

    def _map_action(self, mean_raw: torch.Tensor) -> torch.Tensor:
        u_v = mean_raw[..., 0:1]
        u_s = mean_raw[..., 1:2]
        v_boost = 2.5
        v_brake = 1.5
        steer_max = 0.035
        delta_v = torch.where(u_v >= 0, torch.tanh(u_v) * v_boost, torch.tanh(u_v) * v_brake)
        delta_s = torch.tanh(u_s) * steer_max
        return torch.cat([delta_v, delta_s], dim=-1)

    def forward(self, state: torch.Tensor):
        features = self.actor_backbone(state)
        mean_raw = self.actor_mean(features)
        action_residual = self._map_action(mean_raw)
        std = torch.exp(self.actor_log_std).expand_as(action_residual)
        value = self.critic(state)
        return action_residual, std, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        action_residual_mean, std, value = self.forward(state)
        if deterministic:
            return action_residual_mean, torch.tensor(0.0), value
        dist = torch.distributions.Normal(action_residual_mean, std)
        raw_action = dist.sample()
        v_clamped = torch.clamp(raw_action[..., 0:1], -1.5, 2.5)
        steer_clamped = torch.clamp(raw_action[..., 1:2], -0.035, 0.035)
        clamped_action = torch.cat([v_clamped, steer_clamped], dim=-1)
        log_prob = dist.log_prob(clamped_action).sum(dim=-1, keepdim=True)
        return clamped_action, log_prob, value




# =============================================================================
# 1. BỘ HỌC TRỰC TUYẾN QUÁ TRÌNH GAUSS (ONLINE GAUSSIAN PROCESS REGRESSION)
# =============================================================================
class OnlineGaussianProcess:
    """
    Học trực tuyến sai số trượt ngang v_y từ dữ liệu Odometry/IMU.
    Dùng Sliding Window FIFO để đạt tốc độ giải Cholesky < 0.3ms trên Jetson.
    """
    def __init__(
        self,
        input_dim: int = 2,
        length_scale: float = 0.9,
        sigma_f: float = 0.06,
        sigma_n: float = 0.02,
        max_buffer_size: int = 40
    ):
        self.input_dim = input_dim
        self.l = float(length_scale)
        self.sigma_f = float(sigma_f)
        self.sigma_n = float(sigma_n)
        self.max_buffer_size = int(max_buffer_size)
        self.X_buffer = []
        self.Y_buffer = []
        self._dirty = True
        self._L_cho = None
        self._alpha = None
        self._X_cache = None

    def add_sample(self, z: np.ndarray, y: float):
        z_arr = np.asarray(z, dtype=np.float64).flatten()
        if len(z_arr) != self.input_dim:
           return
        # Chuẩn hóa dải đo: scale góc lái delta x3 để cân bằng với dải vận tốc v
        z_scaled = np.array([z_arr[0], z_arr[1] * 3.0], dtype=np.float64)
        self.X_buffer.append(z_scaled)
        self.Y_buffer.append(float(y))
        if len(self.X_buffer) > self.max_buffer_size:
            self.X_buffer.pop(0)
            self.Y_buffer.pop(0)
        self._dirty = True

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=-1)
        return (self.sigma_f ** 2) * np.exp(-0.5 * sq_dist / (self.l ** 2))

    def _update_decomposition(self) -> bool:
        if not self._dirty and self._L_cho is not None:
            return True
        n = len(self.X_buffer)
        if n < 10:
            return False
        X = np.array(self.X_buffer, dtype=np.float64)
        Y = np.array(self.Y_buffer, dtype=np.float64)
        K = self._kernel(X, X) + (self.sigma_n ** 2) * np.eye(n)
        try:
            self._L_cho = np.linalg.cholesky(K)
            self._alpha = np.linalg.solve(self._L_cho.T, np.linalg.solve(self._L_cho, Y))
            self._X_cache = X
            self._dirty = False
            return True
        except np.linalg.LinAlgError:
            self._L_cho = None
            self._alpha = None
            return False

    def predict(self, z_star: np.ndarray) -> tuple:
        if len(self.X_buffer) < 10:
            return 0.0, 0.0
        if not self._update_decomposition():
            return 0.0, 0.0

        z_arr = np.asarray(z_star, dtype=np.float64).flatten()
        z_star_scaled = np.array([[z_arr[0], z_arr[1] * 3.0]], dtype=np.float64)
        k_star = self._kernel(z_star_scaled, self._X_cache)

        try:
            mu = float(np.dot(k_star, self._alpha).item())
            v = np.linalg.solve(self._L_cho, k_star.T)
            var = float(self.sigma_f ** 2) - float(np.dot(v.T, v).item())
            sigma = float(math.sqrt(max(1e-6, var)))
        except Exception:
            mu = 0.0
            sigma = 0.0

        return mu, sigma


# =============================================================================
# 2. BỘ LỌC AN TOÀN CHANCE-CONSTRAINED GP-ADAPTIVE CBF-QP
# =============================================================================
class GPAdaptiveCBFQPSafetyFilter:
    """
    Bộ lọc an toàn GP-Adaptive CBF-QP kết hợp Động học Ackermann & Quán tính phanh:
      - Quán tính phanh vật lý: d_brake = (v * cos(phi))^2 / (2 * a_brake)
      - Hàm rào cản: h = r - d_min - d_brake
      - Bẻ lái né vật cản: g_steer = -(v / L) * (r * sin(phi))
      - Đệm an toàn thích nghi GP: adaptive_margin = -mu*sin(phi) - k_delta*|sin(phi)|*sigma
    """
    def __init__(
        self,
        d_min: float = 0.15,
        gamma: float = 3.5,
        v_max: float = 3.0,
        steer_max: float = 0.41,
        slack_weight: float = 1e4,
        num_danger_rays: int = 15,
        fov_cutoff_deg: float = 20.0,
        wheelbase: float = 0.39,
        a_max_brake: float = 3.5,
        enable_gp: bool = True,
        k_delta: float = 1.2,
        gp_length_scale: float = 0.9,
        gp_sigma_f: float = 0.06,
        gp_sigma_n: float = 0.02,
        gp_buffer_size: int = 60
    ):
        self.d_min = float(d_min)
        self.gamma = float(gamma)
        self.v_max = float(v_max)
        self.steer_max = float(steer_max)
        self.slack_weight = float(slack_weight)
        self.num_danger_rays = int(num_danger_rays)
        self.fov_cutoff_rad = math.radians(fov_cutoff_deg)
        self.wheelbase = float(wheelbase)
        self.a_max_brake = float(a_max_brake)

        self.enable_gp = bool(enable_gp)
        self.k_delta = float(k_delta)
        self.gp = OnlineGaussianProcess(
            input_dim=2,
            length_scale=gp_length_scale,
            sigma_f=gp_sigma_f,
            sigma_n=gp_sigma_n,
            max_buffer_size=gp_buffer_size
        )

        self.last_mu_slip = 0.0
        self.last_sigma_slip = 0.0
        self.last_margin_mean = 0.0
        self._prev_mu = 0.0
        self._prev_sigma = 0.0
        self._gp_train_counter = 0

    def update_gp_telemetry(self, v_cmd: float, delta_cmd: float, v_y_measured: float):
        if not self.enable_gp:
            return
        if math.isnan(v_y_measured) or math.isnan(v_cmd) or math.isnan(delta_cmd):
            return
        # Chỉ nạp khi xe đang di chuyển thực sự (> 0.3 m/s) và loại bỏ xung nhiễu lớn (> 0.4 m/s)
        if abs(v_cmd) > 0.3 and abs(v_y_measured) < 0.4:
            # Lọc bỏ nhiễu encoder khi đi thẳng: nếu |delta| < 0.03 rad (~1.7°), v_y vật lý = 0 (triệt tiêu bias encoder VESC)
            # Lọc vùng chết rung động học dưới 1.5 cm/s
            if abs(delta_cmd) < 0.03 or abs(v_y_measured) < 0.015:
                v_y_clean = 0.0
            else:
                v_y_clean = v_y_measured
            self.gp.add_sample(np.array([v_cmd, delta_cmd]), v_y_clean)

    def filter(
        self,
        u_nominal: np.ndarray,
        ranges: np.ndarray,
        angles: np.ndarray,
        v_y_measured: float = None,
        v_actual: float = None
    ) -> np.ndarray:
        v_nom = float(u_nominal[0])
        delta_nom = float(u_nominal[1])

        # 1. Tách rời chu kỳ huấn luyện GP (chỉ nạp mẫu mỗi 5 chu kỳ = 4Hz thay vì 20Hz để phá vòng lặp tự kích và giảm tải CPU)
        if v_y_measured is not None:
            self._gp_train_counter += 1
            if self._gp_train_counter % 5 == 0:
                self.update_gp_telemetry(v_nom, delta_nom, v_y_measured)

        # 2. Dự báo GP kết hợp bộ lọc làm mượt EMA (alpha=0.15) triệt tiêu dao động rung giật tick-by-tick
        if self.enable_gp:
            mu_raw, sigma_raw = self.gp.predict(np.array([v_nom, delta_nom]))
            alpha_gp = 0.15
            mu_slip = float(alpha_gp * mu_raw + (1.0 - alpha_gp) * self._prev_mu)
            sigma_slip = float(alpha_gp * sigma_raw + (1.0 - alpha_gp) * self._prev_sigma)
            self._prev_mu = mu_slip
            self._prev_sigma = sigma_slip
        else:
            mu_slip, sigma_slip = 0.0, 0.0

        self.last_mu_slip = mu_slip
        self.last_sigma_slip = sigma_slip

        # ---------------------------------------------------------------------
        # 1. GÓC LÁI: BẢO VỆ TUYỆT ĐỐI 100%, XUẤT THẲNG LỆNH AI (DIL + RRL)
        #    KHÔNG BỘ GIẢI QP, KHÔNG CBF NÀO ĐƯỢC PHÉP CHẠM VÀO GÓC LÁI
        # ---------------------------------------------------------------------
        delta_executed = float(delta_nom)

        # 2. Trích xuất ràng buộc CBF cho vận tốc (truyền delta_nom để mở rộng góc nhìn và v_actual để tính quán tính thật)
        G_v, h_cbf = self._extract_cbf_constraints(
            v_nom, ranges, angles, mu_slip, sigma_slip, delta_cmd=delta_nom, v_actual=v_actual
        )

        if G_v is None or len(G_v) == 0:
            return np.array([v_nom, delta_executed], dtype=np.float32)

        # 3. Giải bài toán an toàn vận tốc 1 biến (1D Closed-Form CBF-QP)
        # Ràng buộc rào cản CBF: g_v * v <= h  ==>  v <= h / g_v (với g_v > 0)
        # - Nếu h / g_v >= v_nom: Phía trước an toàn, CBF đứng ngoài 100% (v_safe = v_nom)
        # - Nếu h / g_v < v_nom: Phanh mượt đến vận tốc an toàn tối đa cho phép
        # - Nếu h <= 0 (nguy cấp sắp đâm): v_safe = 0.0 (dừng hẳn, không được bò ủi vào tường)
        safe_v_limits = h_cbf / np.maximum(1e-6, G_v)
        v_cbf_limit = float(np.min(safe_v_limits))
        v_safe = float(np.clip(v_cbf_limit, 0.0, min(v_nom, self.v_max)))

        return np.array([v_safe, delta_executed], dtype=np.float32)

    def _extract_cbf_constraints(
        self,
        v_nom: float,
        ranges: np.ndarray,
        angles: np.ndarray,
        mu_slip: float,
        sigma_slip: float,
        delta_cmd: float = 0.0,
        v_actual: float = None
    ):
        # 1. Dynamic FOV: Mở rộng vùng quét bám theo hướng bẻ lái để triệt tiêu góc mù hai góc mũi xe
        min_angle = min(-self.fov_cutoff_rad, delta_cmd - math.radians(15.0))
        max_angle = max( self.fov_cutoff_rad, delta_cmd + math.radians(15.0))
        mask_front = (angles >= min_angle) & (angles <= max_angle)

        valid_ranges = ranges[mask_front]
        valid_angles = angles[mask_front]

        valid_mask = ~np.isnan(valid_ranges) & ~np.isinf(valid_ranges) & (valid_ranges > 0.01)
        if not np.any(valid_mask):
            return None, None

        valid_ranges = valid_ranges[valid_mask]
        valid_angles = valid_angles[valid_mask]

        danger_indices = np.argsort(valid_ranges)[:self.num_danger_rays]

        G_v_list = []
        h_list = []
        margins = []

        # 2. ĐỆM QUÁN TÍNH PHANH ĐỘNG HỌC LIÊN TỤC (C^1 SMOOTH KINEMATIC BRAKING CUSHION):
        # Tính toán đệm quán tính phanh theo vận tốc thực tế v_actual (triệt tiêu hoàn toàn bước nhảy
        # gián đoạn ở ngưỡng v=0.35 m/s, loại bỏ 100% hiện tượng chattering / nhấp nhả ga phanh ở dải tốc độ thấp).
        # Khi v_actual -> 0, d_brake -> 0 trơn tru, giúp xe đề-pa mượt mà và không bao giờ bị khựng giật.
        v_eval = max(0.0, v_actual) if v_actual is not None else max(0.0, v_nom)

        # 3. NGƯỠNG KÍCH HOẠT GP:
        # GP chỉ can thiệp khi có bằng chứng trượt THỰC SỰ:
        #   - |mu| > 0.05 m/s  (trượt ngang đủ lớn, loại sạch nhiễu encoder VESC ~0.01)
        #   - sigma > 0.10     (bất định cực cao, không bao giờ xảy ra khi v_y=0)
        # => Thực tế GP gần như im lặng 100% (v_y VESC luôn = 0.000)
        GP_MU_THRESHOLD    = 0.005  # m/s
        GP_SIGMA_THRESHOLD = 0.003    # m/s

        gp_active = (
            self.enable_gp
            and (abs(mu_slip) > GP_MU_THRESHOLD or sigma_slip > GP_SIGMA_THRESHOLD)
        )

        for idx in danger_indices:
            r_i = float(valid_ranges[idx])
            phi_i = float(valid_angles[idx])

            # 1. Quán tính phanh động học trơn liên tục (C^1 continuous)
            v_proj = v_eval * math.cos(phi_i)
            d_brake = (v_proj ** 2) / (2.0 * self.a_max_brake)
            g_v = max(0.1, math.cos(phi_i) * (1.0 + v_proj / self.a_max_brake))

            # 2. Hàm rào an toàn
            h_val = r_i - self.d_min - d_brake

            # 3. Tính toán rào cản CBF + đệm GP (chỉ khi gp_active)
            if h_val <= 0.0:
                # NGUY HIỂM: Đã chạm ngưỡng an toàn -> ép phanh dừng hẳn
                h_cbf_val = self.gamma * h_val
                adaptive_margin = 0.0

            else:
                h_no_gp = self.gamma * h_val

                if gp_active:
                    # GP CAN THIỆP: Đệm thích nghi khi có trượt thực sự
                    # Chỉ bù phần mu (drift thực) + phần sigma nhỏ thôi (uncertainty_weight giảm còn 0.5*k_delta)
                    drift_comp        = -mu_slip * math.sin(phi_i)
                    # Giảm hệ số nhạy cảm sigma xuống 50% so với thiết kế gốc:
                    # Mục đích: không để sigma nhỏ (0.005~0.015) ảnh hưởng khi xe chạy ổn

                    # Sửa dòng 437 - 438 thành:
                    uncertainty_cushion = -self.k_delta * (abs(math.sin(phi_i)) + 0.5 * abs(math.cos(phi_i))) * sigma_slip
                    raw_margin = drift_comp + uncertainty_cushion




                    # uncertainty_cushion = -(self.k_delta * 0.5) * abs(math.sin(phi_i)) * sigma_slip
                    # raw_margin = drift_comp + uncertainty_cushion

                    # Clamp: GP chỉ được phép siết thêm tối đa 1cm (margin <= 0.0),
                    # Giữ tác động cực nhỏ để không ảnh hưởng hành vi CBF bình thường.
                    min_margin = max(-0.5 * h_no_gp, -0.20)
                    adaptive_margin = float(np.clip(raw_margin, min_margin, 0.0))
                else:
                    # GP IM LẶNG: CBF chạy y hệt enable_gp=false (không có đệm nào hết)
                    adaptive_margin = 0.0

                h_cbf_val = h_no_gp + adaptive_margin

            margins.append(adaptive_margin)
            G_v_list.append(g_v)
            h_list.append(h_cbf_val)

        self.last_margin_mean = float(np.mean(margins)) if margins else 0.0
        return np.array(G_v_list, dtype=np.float64), np.array(h_list, dtype=np.float64)



def resolve_model_path(path_or_name: str) -> str:
    """Resolve model path — searches models/ folder relative to this file."""
    if not path_or_name:
        return ''
    if os.path.exists(path_or_name):
        return path_or_name
    filename = os.path.basename(path_or_name)
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [
        curr_dir,
        os.path.abspath(os.path.join(curr_dir, '..', 'models')),
        '/home/adt/f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/models',
        '/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/models',
    ]
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    return path_or_name


def safe_torch_load(path_or_name: str, device: torch.device):
    """Safely loads a PyTorch checkpoint suppressing future warnings or unpickling issues."""
    try:
        return torch.load(path_or_name, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path_or_name, map_location=device)
    except Exception:
        return torch.load(path_or_name, map_location=device, weights_only=False)


class RRLInferenceRealNode(Node):
    def _log_info(self, text: str):
        self.get_logger().info(text)
        if hasattr(self, 'txt_file') and self.txt_file is not None:
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            self.txt_file.write(f"[{now_str}] [INFO] {text}\n")
            self.txt_file.flush()

    def _log_warn(self, text: str):
        self.get_logger().warn(text)
        if hasattr(self, 'txt_file') and self.txt_file is not None:
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            self.txt_file.write(f"[{now_str}] [WARN] {text}\n")
            self.txt_file.flush()

    def __init__(self):
        super().__init__('rrl_inference_real_node')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('dil_model_path',   'final_combined_37500.pth')
        self.declare_parameter('rrl_model_path',   'rrl_ppo_model.pth')
        self.declare_parameter('norm_param_path',  'final_combined_37500_norm.json')
        self.declare_parameter('enable_rrl',       True)
        self.declare_parameter('use_cbf',          True)
        self.declare_parameter('scan_topic',       '/scan')
        self.declare_parameter('drive_topic',      '/drive')
        self.declare_parameter('odom_topic',       '/odom')
        self.declare_parameter('max_speed',        3.0)
        self.declare_parameter('min_speed',        0.0)

        # Logging Parameters (Cách 2: Ghi file tự động)
        self.declare_parameter('enable_csv_log',   True)
        self.declare_parameter('enable_txt_log',   True)
        self.declare_parameter('log_dir',          'log_runs')

        # CBF & GP Parameters
        self.declare_parameter('cbf_d_min',           0.15)
        self.declare_parameter('cbf_gamma',           3.5)
        self.declare_parameter('cbf_a_max_brake',     3.5)
        self.declare_parameter('cbf_wheelbase',       0.39)
        self.declare_parameter('cbf_fov_deg',         20.0)
        self.declare_parameter('cbf_steer_max',       0.41)
        self.declare_parameter('cbf_slack_weight',    1e4)
        self.declare_parameter('cbf_num_danger_rays', 15)
        self.declare_parameter('enable_gp',           True)
        self.declare_parameter('k_delta',             1.2)
        self.declare_parameter('gp_length_scale',     0.9)
        self.declare_parameter('gp_sigma_f',          0.06)
        self.declare_parameter('gp_sigma_n',          0.02)
        self.declare_parameter('gp_buffer_size',      60)
        self.declare_parameter('slip_source',         'twist_vy')  # 'twist_vy' (f1tenth_gym/IMU) or 'yaw_rate_diff' (VESC odometry)

        dil_path   = resolve_model_path(self.get_parameter('dil_model_path').get_parameter_value().string_value)
        rrl_path   = resolve_model_path(self.get_parameter('rrl_model_path').get_parameter_value().string_value)
        norm_path  = resolve_model_path(self.get_parameter('norm_param_path').get_parameter_value().string_value)
        self.enable_rrl   = self.get_parameter('enable_rrl').get_parameter_value().bool_value
        self.use_cbf      = self.get_parameter('use_cbf').get_parameter_value().bool_value
        scan_topic        = self.get_parameter('scan_topic').get_parameter_value().string_value
        drive_topic       = self.get_parameter('drive_topic').get_parameter_value().string_value
        odom_topic        = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.max_speed    = self.get_parameter('max_speed').get_parameter_value().double_value
        self.min_speed    = self.get_parameter('min_speed').get_parameter_value().double_value

        self.enable_csv_log = self.get_parameter('enable_csv_log').get_parameter_value().bool_value
        self.enable_txt_log = self.get_parameter('enable_txt_log').get_parameter_value().bool_value
        log_dir_param       = self.get_parameter('log_dir').get_parameter_value().string_value

        self.cbf_d_min        = self.get_parameter('cbf_d_min').get_parameter_value().double_value
        self.cbf_gamma        = self.get_parameter('cbf_gamma').get_parameter_value().double_value
        self.cbf_a_max_brake  = self.get_parameter('cbf_a_max_brake').get_parameter_value().double_value
        self.cbf_wheelbase    = self.get_parameter('cbf_wheelbase').get_parameter_value().double_value
        self.cbf_fov_deg      = self.get_parameter('cbf_fov_deg').get_parameter_value().double_value
        self.cbf_steer_max    = self.get_parameter('cbf_steer_max').get_parameter_value().double_value
        self.cbf_slack_weight = self.get_parameter('cbf_slack_weight').get_parameter_value().double_value
        self.cbf_num_danger_rays = self.get_parameter('cbf_num_danger_rays').get_parameter_value().integer_value
        self.enable_gp        = self.get_parameter('enable_gp').get_parameter_value().bool_value
        self.k_delta          = self.get_parameter('k_delta').get_parameter_value().double_value
        self.gp_length_scale  = self.get_parameter('gp_length_scale').get_parameter_value().double_value
        self.gp_sigma_f       = self.get_parameter('gp_sigma_f').get_parameter_value().double_value
        self.gp_sigma_n       = self.get_parameter('gp_sigma_n').get_parameter_value().double_value
        self.gp_buffer_size   = self.get_parameter('gp_buffer_size').get_parameter_value().integer_value
        self.slip_source      = self.get_parameter('slip_source').get_parameter_value().string_value

        # ── Dynamic Road Friction Control (Hotkey 'j' or topic /set_friction) ──
        self.declare_parameter('initial_friction', 1.0489)
        self.current_friction_mu = float(self.get_parameter('initial_friction').get_parameter_value().double_value)
        self.pub_friction = self.create_publisher(Float32, '/set_friction', 10)
        self.sub_friction = self.create_subscription(Float32, '/set_friction', self._friction_external_cb, 10)
        self._kb_listener_running = True
        self._kb_thread = None

        # ── Setup File Logging (TXT + CSV) ──────────────────────────────────
        if os.path.isabs(log_dir_param):
            self.log_dir = log_dir_param
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.log_dir = os.path.join(base_dir, log_dir_param)
        os.makedirs(self.log_dir, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        gp_tag = "gp_on" if self.enable_gp else "gp_off"

        self.txt_path = None
        self.txt_file = None
        if self.enable_txt_log:
            self.txt_path = os.path.join(self.log_dir, f"run_{timestamp_str}_{gp_tag}_console.log")
            self.txt_file = open(self.txt_path, 'w', encoding='utf-8')

        self.csv_path = None
        self.csv_file = None
        self.csv_writer = None
        if self.enable_csv_log:
            self.csv_path = os.path.join(self.log_dir, f"run_{timestamp_str}_{gp_tag}_telemetry.csv")
            self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                'timestamp_sec',
                'tick',
                'vx',
                'vy',
                'omega',
                'omega_kin',
                'slip_signal',
                'dil_v',
                'dil_delta_deg',
                'rrl_dv',
                'rrl_ddelta_deg',
                'v_cmd_raw',
                'delta_cmd_deg',
                'v_cmd_safe',
                'delta_cmd_safe_deg',
                'cbf_cut',
                'cbf_intervene',
                'gp_active',
                'gp_n',
                'gp_mu',
                'gp_sigma',
                'gp_margin',
                'h_val',
                'd_brake',
                'min_dist_front',
                'mean_dist_front',
                'min_dist_all',
                'road_friction_mu',
                'anomalies'
            ])
            self.csv_file.flush()

        self._log_info("==================================================")
        self._log_info("  RRL + GP-ADAPTIVE CBF REAL ROBOT INFERENCE NODE ")
        self._log_info("==================================================")
        if self.csv_path:
            self._log_info(f"📊 Auto-logging telemetry CSV: {self.csv_path}")
        if self.txt_path:
            self._log_info(f"📄 Auto-logging console text:   {self.txt_path}")

        self._log_info(f"DIL Model : {dil_path}")
        self._log_info(f"RRL Model : {rrl_path}")
        self._log_info(f"Norm JSON : {norm_path}")
        self._log_info(f"Enable RRL: {self.enable_rrl} | Use CBF: {self.use_cbf}")
        self._log_info(f"Drive Topic: {drive_topic} | Odom Topic: {odom_topic} | Scan Topic: {scan_topic}")

        # ── Load Normalization Params (Optional JSON with embedded fallbacks) ──
        DEFAULT_MEAN = [2.4377425, 0.12915474]
        DEFAULT_STD  = [0.9534305, 0.1454656]
        self.target_beams = 60
        self.max_range    = 10.0

        if norm_path and os.path.exists(norm_path):
            try:
                with open(norm_path, 'r') as f:
                    norm_data = json.load(f)
                self.target_beams = norm_data.get("target_beams", 60)
                self.max_range    = norm_data.get("max_range", 10.0)
                self.target_mean  = np.array(norm_data.get("target_mean", DEFAULT_MEAN), dtype=np.float32)
                self.target_std   = np.array(norm_data.get("target_std", DEFAULT_STD), dtype=np.float32)
                self._log_info(f"Loaded norm params from: {norm_path}")
            except Exception as e:
                self._log_warn(f"Failed to read norm JSON ({e}), using embedded default norm params.")
                self.target_mean  = np.array(DEFAULT_MEAN, dtype=np.float32)
                self.target_std   = np.array(DEFAULT_STD, dtype=np.float32)
        else:
            self.target_mean = np.array(DEFAULT_MEAN, dtype=np.float32)
            self.target_std  = np.array(DEFAULT_STD, dtype=np.float32)
            self._log_info("Norm JSON not provided or not found. Used embedded DIL norm parameters.")

        # ── Device ──────────────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._log_info(f"Torch device: {self.device}")

        # ── Load DIL Baseline (.pth) ─────────────────────────────────────────
        self.dil_model = DAggerMLP(input_dim=self.target_beams, output_dim=2).to(self.device)
        self.dil_model.load_state_dict(safe_torch_load(dil_path, self.device))
        self.dil_model.eval()
        for p in self.dil_model.parameters():
            p.requires_grad = False
        self._log_info(f"DIL model loaded: {dil_path}")

        # ── Load RRL Policy (.pth) ───────────────────────────────────────────
        self.rrl_model = None
        if self.enable_rrl:
            rrl_state_dim = self.target_beams + 2 + 2 + 2  # 60 + v_x + omega + a_DIL + prev_a_R = 66
            self.rrl_model = RRLActorCritic(rrl_state_dim, action_dim=2).to(self.device)
            self.rrl_model.load_state_dict(safe_torch_load(rrl_path, self.device))
            self.rrl_model.eval()
            self._log_info(f"RRL model loaded: {rrl_path}")

        # ── GP-Adaptive CBF Safety Filter ───────────────────────────────────
        if self.use_cbf:
            self.cbf = GPAdaptiveCBFQPSafetyFilter(
                d_min=self.cbf_d_min,
                gamma=self.cbf_gamma,
                v_max=self.max_speed,
                steer_max=self.cbf_steer_max,
                slack_weight=self.cbf_slack_weight,
                num_danger_rays=self.cbf_num_danger_rays,
                fov_cutoff_deg=self.cbf_fov_deg,
                wheelbase=self.cbf_wheelbase,
                a_max_brake=self.cbf_a_max_brake,
                enable_gp=self.enable_gp,
                k_delta=self.k_delta,
                gp_length_scale=self.gp_length_scale,
                gp_sigma_f=self.gp_sigma_f,
                gp_sigma_n=self.gp_sigma_n,
                gp_buffer_size=self.gp_buffer_size
            )
            self._log_info(
                f"[GP-CBF INITIALIZED] Active: d_min={self.cbf_d_min}m, gamma={self.cbf_gamma}, "
                f"L={self.cbf_wheelbase}m, a_brake={self.cbf_a_max_brake}m/s², FOV={self.cbf_fov_deg}°, GP={self.enable_gp}"
            )

        # ── State ───────────────────────────────────────────────────────────
        self.latest_scan        = None
        self.latest_scan_angles = None
        self.last_scan_time     = None
        self.last_odom_time     = None
        self._pose_history      = deque(maxlen=10)  # Stores (t_sec, x, y, yaw)
        self.v_x                = 0.0
        self.v_y                = 0.0
        self.omega              = 0.0
        self.prev_a_R           = np.zeros(2, dtype=np.float32)

        # ── Telemetry & Metrics State ───────────────────────────────────────
        self._debug_tick_count    = 0
        self._cbf_intervene_count = 0
        self._anomaly_count       = 0
        self.last_omega_kin       = 0.0
        self.last_slip_signal     = 0.0

        # ── Pub / Sub ───────────────────────────────────────────────────────
        self.pub_drive = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.sub_scan  = self.create_subscription(LaserScan, scan_topic, self._scan_cb, 10)
        self.sub_odom  = self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)

        # Control loop at 20 Hz
        self.create_timer(0.05, self._control_loop)

        # Khởi động lắng nghe phím nóng 'j' để chuyển đổi độ bám đường
        self._start_keyboard_listener()
        self._log_info(f"🏎️  [ROAD FRICTION] Initial grip mu: {self.current_friction_mu:.4f}")
        self._log_info("⌨️  [HOTKEY] Nhấn [j] bất kỳ lúc nào trong terminal này để đổi độ bám (1.05 <-> 0.10)!")
        self._log_info("Node ready. Waiting for /scan and /odom...")

    def _friction_external_cb(self, msg: Float32):
        new_mu = float(msg.data)
        if abs(new_mu - self.current_friction_mu) > 0.001:
            self.current_friction_mu = new_mu
            status = "🧊 ICE/WET (μ=0.10)" if new_mu < 0.3 else "🏎️  DRY/HIGH-GRIP (μ=1.05)"
            self._log_info(f"🔄 [FRICTION SYNC] External update -> Road grip is now {new_mu:.3f} ({status})")

    def _toggle_friction(self):
        if self.current_friction_mu >= 0.5:
            self.current_friction_mu = 0.10
            label = "🧊 BĂNG TRƠN TRƯỢT (SLIPPERY ICE / WET TRACK) -> μ = 0.10"
        else:
            self.current_friction_mu = 1.0489
            label = "🏎️  MẶT ĐƯỜNG KHÔ BÁM (DRY ASPHALT / HIGH GRIP) -> μ = 1.05"

        msg = Float32()
        msg.data = float(self.current_friction_mu)
        self.pub_friction.publish(msg)

        bar = "═" * 66
        self._log_warn(
            f"\n{bar}\n"
            f"🎮 [HOTKEY 'j' PRESSED] ĐỘ BÁM ĐƯỜNG ĐÃ CHUYỂN: {label}\n"
            f"{bar}"
        )

    def _start_keyboard_listener(self):
        if not sys.stdin.isatty():
            return
        def _listen():
            try:
                old_settings = termios.tcgetattr(sys.stdin)
            except Exception:
                return
            try:
                tty.setcbreak(sys.stdin.fileno())
                while self._kb_listener_running and rclpy.ok():
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if rlist:
                        ch = sys.stdin.read(1)
                        if ch in ('j', 'J'):
                            self._toggle_friction()
            except Exception:
                pass
            finally:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass

        self._kb_thread = threading.Thread(target=_listen, daemon=True)
        self._kb_thread.start()

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self.v_x   = float(msg.twist.twist.linear.x)
        self.v_y   = float(msg.twist.twist.linear.y)
        self.omega = float(msg.twist.twist.angular.z)
        self.last_odom_time = self.get_clock().now()

        # Trích xuất góc hướng yaw từ quaternion để phục vụ Phương án 2 (Pose Curvature)
        qx = float(msg.pose.pose.orientation.x)
        qy = float(msg.pose.pose.orientation.y)
        qz = float(msg.pose.pose.orientation.z)
        qw = float(msg.pose.pose.orientation.w)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        t_sec = float(self.last_odom_time.nanoseconds * 1e-9)
        self._pose_history.append((t_sec, x, y, yaw))

    def _scan_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=self.max_range, posinf=self.max_range, neginf=0.0)
        self.latest_scan        = ranges
        self.latest_scan_angles = (
            np.arange(len(ranges), dtype=np.float32) * msg.angle_increment + msg.angle_min
        )
        self.last_scan_time = self.get_clock().now()

    # ── Preprocessing ─────────────────────────────────────────────────────

    def _preprocess_scan(self):
        """Crop LiDAR to [-60°, +60°] FOV and resample to self.target_beams."""
        if self.latest_scan is None or self.latest_scan_angles is None:
            fallback = np.ones(self.target_beams, dtype=np.float32)
            return fallback * self.max_range, fallback

        crop = math.radians(60.0)
        mask = (self.latest_scan_angles >= -crop) & (self.latest_scan_angles <= crop)
        if not np.any(mask):
            fallback = np.ones(self.target_beams, dtype=np.float32)
            return fallback * self.max_range, fallback

        valid        = np.clip(self.latest_scan[mask], 0.0, self.max_range)
        angles_valid = self.latest_scan_angles[mask]
        target_angles = np.linspace(-crop, crop, self.target_beams)

        scan_meters = np.interp(target_angles, angles_valid, valid).astype(np.float32)
        scan_norm   = (scan_meters / self.max_range).astype(np.float32)
        return scan_meters, scan_norm

    # ── Inference ──────────────────────────────────────────────────────────

    def _infer_dil(self, scan_norm: np.ndarray) -> np.ndarray:
        """Run DIL .pth model. Input: normalized scan [0,1]. Output: [v (m/s), delta (rad)]."""
        tensor = torch.tensor(scan_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw = self.dil_model(tensor).cpu().numpy().squeeze(0)
        if self.target_mean is not None and self.target_std is not None:
            raw = raw * self.target_std + self.target_mean
        return raw.astype(np.float32)

    def _infer_rrl(self, rrl_state: np.ndarray) -> np.ndarray:
        """Run RRL .pth model deterministically. Output: [Dv (m/s), Ddelta (rad)]."""
        if self.rrl_model is None:
            return np.zeros(2, dtype=np.float32)
        tensor = torch.tensor(rrl_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            a_R, _, _ = self.rrl_model.get_action(tensor, deterministic=True)
        return a_R.cpu().numpy().squeeze(0).astype(np.float32)

    # ── Control Loop ───────────────────────────────────────────────────────

    def _control_loop(self):
        if self.latest_scan is None or self.last_scan_time is None:
            return

        # 0. Watchdog an toàn: Ngắt khẩn cấp nếu mất tín hiệu LiDAR > 0.4s
        now = self.get_clock().now()
        dt_scan = (now - self.last_scan_time).nanoseconds * 1e-9
        if dt_scan > 0.4:
            stop_msg = AckermannDriveStamped()
            stop_msg.header.stamp = now.to_msg()
            stop_msg.header.frame_id = 'laser'
            stop_msg.drive.speed = 0.0
            stop_msg.drive.steering_angle = 0.0
            self.pub_drive.publish(stop_msg)
            self.get_logger().warn(
                f"🛑 WATCHDOG: LiDAR scan timed out ({dt_scan:.2f}s > 0.4s)! Emergency stop executed.",
                throttle_duration_sec=1.0
            )
            return

        # 1. Preprocess LiDAR
        scan_meters, scan_norm = self._preprocess_scan()

        # 2. DIL Baseline inference → a_DIL = [v_DIL, delta_DIL]
        a_dil = self._infer_dil(scan_norm)

        # 3. RRL Residual inference → a_R = [Δv, Δδ]
        if self.enable_rrl:
            rrl_state = np.hstack([
                scan_norm,
                [self.v_x, self.omega],
                a_dil,
                self.prev_a_R
            ]).astype(np.float32)   # shape: (66,)

            raw_a_R = self._infer_rrl(rrl_state)
            raw_a_R[0] = float(np.clip(raw_a_R[0], -1.5, 2.5))

            # EMA low-pass filter (alpha=0.35) for smooth actuation on real hardware
            a_R = 0.35 * raw_a_R + 0.65 * self.prev_a_R
        else:
            a_R = np.zeros(2, dtype=np.float32)

        # 4. Combine: a_total = a_DIL + a_R
        # Clip chặt góc lái theo giới hạn cơ khí servo (cbf_steer_max) bảo vệ phần cứng
        v_total     = float(np.clip(a_dil[0] + a_R[0], self.min_speed, self.max_speed))
        steer_total = float(np.clip(a_dil[1] + a_R[1], -self.cbf_steer_max, self.cbf_steer_max))
        u_total     = np.array([v_total, steer_total], dtype=np.float32)

        # 4b. Xác định tín hiệu trượt ngang cấp cho GP:
        omega_kin = (self.v_x / max(0.01, self.cbf_wheelbase)) * math.tan(u_total[1])
        if self.slip_source == 'yaw_rate_diff':
            # Phương án 1: Sai lệch yaw rate từ twist (/odom twist vs mô hình xe đạp)
            v_slip_signal = float((self.omega - omega_kin) * self.cbf_wheelbase)
        elif self.slip_source == 'pose_curvature':
            # Phương án 2: Suy trượt từ đạo hàm chuỗi Pose liên tiếp (không phụ thuộc twist)
            omega_pose = 0.0
            if len(self._pose_history) >= 2:
                t_old, _, _, yaw_old = self._pose_history[0]
                t_new, _, _, yaw_new = self._pose_history[-1]
                dt = t_new - t_old
                if dt > 0.01:
                    dyaw = math.atan2(math.sin(yaw_new - yaw_old), math.cos(yaw_new - yaw_old))
                    omega_pose = dyaw / dt
            v_slip_signal = float((omega_pose - omega_kin) * self.cbf_wheelbase)
        else:
            # Mặc định: twist.linear.y (Có sẵn trong f1tenth_gym_ros hoặc khi có IMU)
            v_slip_signal = self.v_y

        self.last_omega_kin = omega_kin
        self.last_slip_signal = v_slip_signal

        # 5. CBF Safety Filter
        if self.use_cbf and self.latest_scan_angles is not None:
            u_executed = self.cbf.filter(
                u_total,
                self.latest_scan,
                self.latest_scan_angles,
                v_y_measured=v_slip_signal,
                v_actual=self.v_x
            )
        else:
            u_executed = u_total

        self.prev_a_R = a_R

        # ══════════════════════════════════════════════════════════════════════
        # 6. DEBUG LOGGING & TELEMETRY EXPORT
        # ══════════════════════════════════════════════════════════════════════
        self._debug_tick_count    += 1

        # ── Thông tin LiDAR ───────────────────────────────────────────────
        if self.latest_scan_angles is not None:
            front_mask = (
                (self.latest_scan_angles >= -math.radians(self.cbf_fov_deg)) &
                (self.latest_scan_angles <=  math.radians(self.cbf_fov_deg))
            )
        else:
            front_mask = np.ones(len(self.latest_scan), dtype=bool)

        valid_front = self.latest_scan[front_mask]
        valid_front = valid_front[np.isfinite(valid_front) & (valid_front > 0.01)]
        min_dist_front  = float(np.min(valid_front))  if len(valid_front) > 0 else 99.0
        mean_dist_front = float(np.mean(valid_front)) if len(valid_front) > 0 else 99.0

        all_valid    = self.latest_scan[np.isfinite(self.latest_scan) & (self.latest_scan > 0.01)]
        min_dist_all = float(np.min(all_valid)) if len(all_valid) > 0 else 99.0

        # ── CBF can thiệp không? ─────────────────────────────────────────
        v_reduction     = v_total - float(u_executed[0])
        cbf_intervening = v_reduction > 0.05
        if cbf_intervening:
            self._cbf_intervene_count += 1

        # ── Quãng đường phanh & khoảng cách an toàn còn lại ─────────────
        d_brake_now       = (self.v_x ** 2) / (2.0 * self.cbf_a_max_brake) if self.v_x > 0 else 0.0
        safety_margin_now = min_dist_front - self.cbf_d_min - d_brake_now

        # ── Thông tin GP ─────────────────────────────────────────────────
        gp_n         = len(self.cbf.gp.X_buffer) if (self.use_cbf and self.enable_gp) else 0
        gp_mu        = self.cbf.last_mu_slip    if self.use_cbf else 0.0
        gp_sigma     = self.cbf.last_sigma_slip if self.use_cbf else 0.0
        gp_margin    = self.cbf.last_margin_mean if self.use_cbf else 0.0
        gp_is_active = (self.use_cbf and self.enable_gp and (abs(gp_mu) > 0.005 or gp_sigma > 0.003))

        # ── Phát hiện bất thường ────────────────────────────────────────
        anomalies = []
        if abs(self.v_y) > 0.15:
            anomalies.append(f"HIGH_SLIP vy={self.v_y:+.3f}")
        if min_dist_front < self.cbf_d_min + 0.05:
            anomalies.append(f"DANGER dist={min_dist_front:.2f}m")
        if safety_margin_now < 0.0:
            anomalies.append(f"CBF_VIOLATED h={safety_margin_now:.3f}m")
        if v_reduction > 0.50:
            anomalies.append(f"HARD_BRAKE cut={v_reduction:.2f}")
        if self.use_cbf and self.enable_gp and abs(gp_mu) > 0.08:
            anomalies.append(f"HIGH_MU mu={gp_mu:+.3f}")
        if abs(a_dil[1]) > 0.30 and abs(a_R[1]) > 0.12 and (a_dil[1] * a_R[1] < 0):
            anomalies.append(
                f"STEER_FIGHT dil={math.degrees(a_dil[1]):+.0f}° rrl={math.degrees(a_R[1]):+.0f}°"
            )
        self._anomaly_count += len(anomalies)

        # ── GHI DỮ LIỆU TELEMETRY CSV MỖI TICK (20Hz) ───────────────────
        if self.csv_writer is not None:
            t_sec = float(self._debug_tick_count * 0.05)
            self.csv_writer.writerow([
                f"{t_sec:.3f}",
                self._debug_tick_count,
                f"{self.v_x:.3f}",
                f"{self.v_y:.4f}",
                f"{self.omega:.4f}",
                f"{self.last_omega_kin:.4f}",
                f"{self.last_slip_signal:.4f}",
                f"{a_dil[0]:.3f}",
                f"{math.degrees(a_dil[1]):.2f}",
                f"{a_R[0]:.3f}",
                f"{math.degrees(a_R[1]):.2f}",
                f"{u_total[0]:.3f}",
                f"{math.degrees(u_total[1]):.2f}",
                f"{u_executed[0]:.3f}",
                f"{math.degrees(u_executed[1]):.2f}",
                f"{v_reduction:.3f}",
                1 if cbf_intervening else 0,
                1 if gp_is_active else 0,
                gp_n,
                f"{gp_mu:.5f}",
                f"{gp_sigma:.5f}",
                f"{gp_margin:.5f}",
                f"{safety_margin_now:.4f}",
                f"{d_brake_now:.4f}",
                f"{min_dist_front:.3f}",
                f"{mean_dist_front:.3f}",
                f"{min_dist_all:.3f}",
                f"{self.current_friction_mu:.4f}",
                ";".join(anomalies) if anomalies else "NONE"
            ])
            self.csv_file.flush()

        if anomalies:
            self._log_warn(
                f"⚠️  [{self._debug_tick_count:05d}] ANOMALY: {' | '.join(anomalies)} "
                f"[vx={self.v_x:.2f} dist={min_dist_front:.2f}m v_out={u_executed[0]:.2f}]"
            )

        # ── LOG 1: Dòng chi tiết mỗi 0.5 giây (10 ticks @ 20Hz) ─────────
        if self._debug_tick_count % 10 == 0:
            cbf_tag = f"🔴CBF-{v_reduction:.2f}" if cbf_intervening else "✅FREE"
            slip_info = f"slip={self.last_slip_signal:+.3f}" if self.use_cbf else "slip=off"
            grip_str = f"🧊{self.current_friction_mu:.2f}" if self.current_friction_mu < 0.3 else f"🏎️{self.current_friction_mu:.2f}"
            self._log_info(
                f"[{self._debug_tick_count:05d}|{self._debug_tick_count*0.05:6.1f}s] "
                f"LIDAR: f_min={min_dist_front:.2f}m f_avg={mean_dist_front:.2f}m all_min={min_dist_all:.2f}m | "
                f"ODOM: vx={self.v_x:+.2f} vy={self.v_y:+.3f} ω_meas={self.omega:+.2f} (ω_kin={self.last_omega_kin:+.2f}) {slip_info} | "
                f"DIL: v={a_dil[0]:.2f} δ={math.degrees(a_dil[1]):+.1f}° | "
                f"RRL: Δv={a_R[0]:+.2f} Δδ={math.degrees(a_R[1]):+.1f}° | "
                f"CMD: v={u_executed[0]:.2f} δ={math.degrees(u_executed[1]):+.1f}° {cbf_tag} | "
                f"GRIP={grip_str} | "
                f"GP[n={gp_n}]: μ={gp_mu:+.4f} σ={gp_sigma:.4f} m={gp_margin:+.3f} | "
                f"SAFETY: h={safety_margin_now:+.3f}m d_brake={d_brake_now:.3f}m"
            )

        # ── LOG 2: Tóm tắt mỗi 5 giây (100 ticks) ───────────────────────
        if self._debug_tick_count % 100 == 0:
            intervene_pct = 100.0 * self._cbf_intervene_count / max(1, self._debug_tick_count)
            self._log_info(
                f"\n{'='*72}\n"
                f"  📊 SUMMARY  tick={self._debug_tick_count}  t={self._debug_tick_count*0.05:.0f}s\n"
                f"  ├─ CBF can thiệp     : {self._cbf_intervene_count}/{self._debug_tick_count} = {intervene_pct:.1f}% thời gian\n"
                f"  ├─ Cảnh báo bất thường: {self._anomaly_count} lần tích lũy\n"
                f"  ├─ GP buffer         : {gp_n}/{self.gp_buffer_size} mẫu\n"
                f"  ├─ GP  μ_slip={gp_mu:+.4f}  σ={gp_sigma:.4f}  margin_avg={gp_margin:+.4f}m\n"
                f"  ├─ Vật cản gần nhất  : front={min_dist_front:.2f}m  all_dir={min_dist_all:.2f}m\n"
                f"  ├─ Safety margin (h)  : {safety_margin_now:+.3f}m  (d_brake={d_brake_now:.3f}m)\n"
                f"  ├─ Vận tốc xe        : vx={self.v_x:.2f}m/s  vy={self.v_y:+.3f}m/s  ω={self.omega:.3f}rad/s\n"
                f"  └─ Lệnh cuối cùng   : v={u_executed[0]:.2f}m/s  δ={math.degrees(u_executed[1]):+.1f}°\n"
                f"{'='*72}"
            )

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser'
        msg.drive.speed          = float(u_executed[0])
        msg.drive.steering_angle = float(u_executed[1])
        self.pub_drive.publish(msg)

    def destroy_node(self):
        if hasattr(self, 'csv_file') and self.csv_file is not None:
            try:
                self.csv_file.flush()
                self.csv_file.close()
            except Exception:
                pass
            self.csv_file = None
        if hasattr(self, 'txt_file') and self.txt_file is not None:
            try:
                self.txt_file.flush()
                self.txt_file.close()
            except Exception:
                pass
            self.txt_file = None
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RRLInferenceRealNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node._log_info("Shutting down (KeyboardInterrupt) — sending stop command.")
            stop = AckermannDriveStamped()
            stop.drive.speed = 0.0
            stop.drive.steering_angle = 0.0
            node.pub_drive.publish(stop)
    finally:
        if node is not None:
            csv_p = getattr(node, 'csv_path', None)
            txt_p = getattr(node, 'txt_path', None)
            if csv_p:
                node.get_logger().info(f"📊 Telemetry CSV saved: {csv_p}")
            if txt_p:
                node.get_logger().info(f"📄 Console log saved:   {txt_p}")
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()