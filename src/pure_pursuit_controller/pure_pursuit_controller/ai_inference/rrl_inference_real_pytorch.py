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
  cbf_d_min           : float (default: 0.3)  — Khoảng cách an toàn tối thiểu (m)
  cbf_gamma           : float (default: 3.5)  — Độ dốc hội tụ CBF
  cbf_a_max_brake     : float (default: 2.61) — Gia tốc phanh cực đại (m/s²)
  cbf_wheelbase       : float (default: 0.39) — Chiều dài cơ sở L (m)
  cbf_fov_deg         : float (default: 25.0) — Góc mở radar kiểm soát va chạm (+/- deg)
  cbf_steer_max       : float (default: 0.41) — Giới hạn góc lái tối đa (rad)
  enable_gp           : bool (default: True)  — Bật học thích nghi trượt lốp GP
  k_delta             : float (default: 2.5)  — Hệ số tin cậy Chance-Constraint (2.5 ~ 99.4%)
  gp_buffer_size      : int (default: 40)     — Kích thước bộ nhớ trượt FIFO của GP
  gp_length_scale     : float (default: 0.9)  — Độ nhạy tương đồng vận tốc
  gp_sigma_f          : float (default: 0.6)  — Biên độ trượt lốp tối đa (m/s)
  gp_sigma_n          : float (default: 0.03) — Phương sai nhiễu đo lường Odometry
"""

import os
import json
import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
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


import scipy.optimize as opt
try:
    import scipy.sparse as sp
    _HAS_SCIPY_SPARSE = True
except ImportError:
    _HAS_SCIPY_SPARSE = False

try:
    from qpsolvers import solve_qp
    _HAS_QPSOLVERS = True
except ImportError:
    _HAS_QPSOLVERS = False


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
        sigma_f: float = 0.6,
        sigma_n: float = 0.03,
        max_buffer_size: int = 40
    ):
        self.input_dim = input_dim
        self.l = float(length_scale)
        self.sigma_f = float(sigma_f)
        self.sigma_n = float(sigma_n)
        self.max_buffer_size = int(max_buffer_size)
        self.X_buffer = []
        self.Y_buffer = []

    def add_sample(self, z: np.ndarray, y: float):
        z_arr = np.asarray(z, dtype=np.float64).flatten()
        if len(z_arr) != self.input_dim:
            return
        self.X_buffer.append(z_arr)
        self.Y_buffer.append(float(y))
        if len(self.X_buffer) > self.max_buffer_size:
            self.X_buffer.pop(0)
            self.Y_buffer.pop(0)

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=-1)
        return (self.sigma_f ** 2) * np.exp(-0.5 * sq_dist / (self.l ** 2))

    def predict(self, z_star: np.ndarray) -> tuple:
        z_star = np.asarray(z_star, dtype=np.float64).reshape(1, self.input_dim)
        n = len(self.X_buffer)
        if n < 3:
            return 0.0, float(self.sigma_f)

        X = np.array(self.X_buffer, dtype=np.float64)
        Y = np.array(self.Y_buffer, dtype=np.float64)
        K = self._kernel(X, X) + (self.sigma_n ** 2) * np.eye(n)
        k_star = self._kernel(z_star, X)

        try:
            L_cho = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L_cho.T, np.linalg.solve(L_cho, Y))
            mu = float(np.dot(k_star, alpha).item())
            v = np.linalg.solve(L_cho, k_star.T)
            var = float(self.sigma_f ** 2) - float(np.dot(v.T, v).item())
            sigma = float(math.sqrt(max(1e-6, var)))
        except np.linalg.LinAlgError:
            mu = 0.0
            sigma = float(self.sigma_f)

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
        d_min: float = 0.3,
        gamma: float = 3.5,
        v_max: float = 3.0,
        steer_max: float = 0.41,
        slack_weight: float = 1e4,
        num_danger_rays: int = 15,
        fov_cutoff_deg: float = 25.0,
        wheelbase: float = 0.39,
        a_max_brake: float = 2.61,
        enable_gp: bool = True,
        k_delta: float = 2.5,
        gp_length_scale: float = 0.9,
        gp_sigma_f: float = 0.6,
        gp_sigma_n: float = 0.03,
        gp_buffer_size: int = 40
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
        self.last_sigma_slip = float(gp_sigma_f)
        self.last_margin_mean = 0.0

    def update_gp_telemetry(self, v_cmd: float, delta_cmd: float, v_y_measured: float):
        if not self.enable_gp:
            return
        if math.isnan(v_y_measured) or math.isnan(v_cmd) or math.isnan(delta_cmd):
            return
        if abs(v_cmd) > 0.1:
            self.gp.add_sample(np.array([v_cmd, delta_cmd]), v_y_measured)

    def filter(
        self,
        u_nominal: np.ndarray,
        ranges: np.ndarray,
        angles: np.ndarray,
        v_y_measured: float = None
    ) -> np.ndarray:
        v_nom = float(u_nominal[0])
        delta_nom = float(u_nominal[1])

        if v_y_measured is not None:
            self.update_gp_telemetry(v_nom, delta_nom, v_y_measured)

        if self.enable_gp:
            mu_slip, sigma_slip = self.gp.predict(np.array([v_nom, delta_nom]))
        else:
            mu_slip, sigma_slip = 0.0, 0.0

        self.last_mu_slip = mu_slip
        self.last_sigma_slip = sigma_slip

        G_cbf, h_cbf = self._extract_cbf_constraints(v_nom, ranges, angles, mu_slip, sigma_slip)

        if G_cbf is None or len(G_cbf) == 0:
            return np.array([v_nom, delta_nom], dtype=np.float32)

        if _HAS_QPSOLVERS:
            try:
                P = np.diag([1.0, 5.0, self.slack_weight]).astype(np.float64)
                q = np.array([-v_nom, -5.0 * delta_nom, 0.0], dtype=np.float64)
                lb = np.array([0.0, -self.steer_max, 0.0], dtype=np.float64)
                ub = np.array([self.v_max, self.steer_max, 10.0], dtype=np.float64)

                if _HAS_SCIPY_SPARSE:
                    P_in = sp.csc_matrix(P)
                    G_in = sp.csc_matrix(G_cbf)
                else:
                    P_in = P
                    G_in = G_cbf

                sol = solve_qp(P_in, q, G_in, h_cbf, None, None, lb, ub, solver="osqp")
                if sol is not None:
                    v_safe = float(np.clip(sol[0], 0.0, self.v_max))
                    delta_safe = float(np.clip(sol[1], -self.steer_max, self.steer_max))
                    return np.array([v_safe, delta_safe], dtype=np.float32)
            except Exception:
                pass

        return self._solve_scipy_qp(v_nom, delta_nom, G_cbf, h_cbf)

    def _extract_cbf_constraints(
        self,
        v_nom: float,
        ranges: np.ndarray,
        angles: np.ndarray,
        mu_slip: float,
        sigma_slip: float
    ):
        mask_front = (angles >= -self.fov_cutoff_rad) & (angles <= self.fov_cutoff_rad)
        valid_ranges = ranges[mask_front]
        valid_angles = angles[mask_front]

        valid_mask = ~np.isnan(valid_ranges) & ~np.isinf(valid_ranges) & (valid_ranges > 0.01)
        if not np.any(valid_mask):
            return None, None

        valid_ranges = valid_ranges[valid_mask]
        valid_angles = valid_angles[valid_mask]

        danger_indices = np.argsort(valid_ranges)[:self.num_danger_rays]

        G_list = []
        h_list = []
        margins = []

        v_curr = max(0.5, v_nom)

        for idx in danger_indices:
            r_i = float(valid_ranges[idx])
            phi_i = float(valid_angles[idx])

            # 1. Quán tính phanh động học (Phanh từ xa, không đâm)
            d_brake = (v_curr * math.cos(phi_i))**2 / (2.0 * self.a_max_brake)

            # 2. Hàm rào an toàn
            h_val = r_i - self.d_min - d_brake

            # 3. Đạo hàm theo mô hình Ackermann Kinematics
            g_v = math.cos(phi_i) * (1.0 + (v_curr * math.cos(phi_i)) / self.a_max_brake)
            g_steer = -(v_curr / self.wheelbase) * (r_i * math.sin(phi_i))

            # 4. Đệm thích nghi GP
            if self.enable_gp:
                drift_comp = - mu_slip * math.sin(phi_i)
                uncertainty_cushion = - self.k_delta * abs(math.sin(phi_i)) * sigma_slip
                adaptive_margin = drift_comp + uncertainty_cushion
            else:
                adaptive_margin = 0.0

            margins.append(adaptive_margin)

            G_list.append([g_v, g_steer, -1.0])
            h_list.append(self.gamma * max(0.01, h_val) + adaptive_margin)

        self.last_margin_mean = float(np.mean(margins)) if margins else 0.0
        return np.array(G_list, dtype=np.float64), np.array(h_list, dtype=np.float64)

    def _solve_scipy_qp(self, v_nom: float, delta_nom: float, G: np.ndarray, h: np.ndarray) -> np.ndarray:
        if v_nom <= 0.05:
            return np.array([0.0, delta_nom], dtype=np.float32)

        def objective(x):
            v, steer, slack = x[0], x[1], x[2]
            return 0.5 * (v - v_nom)**2 + 2.5 * (steer - delta_nom)**2 + 0.5 * self.slack_weight * (slack**2)

        def jacobian(x):
            v, steer, slack = x[0], x[1], x[2]
            return np.array([v - v_nom, 5.0 * (steer - delta_nom), self.slack_weight * slack])

        constraints = ({
            'type': 'ineq',
            'fun': lambda x: h - np.dot(G, x),
            'jac': lambda x: -G
        })

        bounds = [
            (0.0, self.v_max),
            (-self.steer_max, self.steer_max),
            (0.0, 10.0)
        ]

        x0 = np.array([
            float(np.clip(v_nom, 0.0, self.v_max)),
            float(np.clip(delta_nom, -self.steer_max, self.steer_max)),
            0.0
        ], dtype=np.float64)

        res = opt.minimize(
            objective,
            x0,
            method='SLSQP',
            jac=jacobian,
            constraints=constraints,
            bounds=bounds,
            options={'ftol': 1e-4, 'maxiter': 25}
        )

        if res.success and res.x is not None:
            v_safe = float(np.clip(res.x[0], 0.0, self.v_max))
            delta_safe = float(np.clip(res.x[1], -self.steer_max, self.steer_max))
            return np.array([v_safe, delta_safe], dtype=np.float32)

        return np.array([max(0.2, min(v_nom * 0.3, self.v_max)), delta_nom], dtype=np.float32)


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
    def __init__(self):
        super().__init__('rrl_inference_real_node')

        self.get_logger().info("==================================================")
        self.get_logger().info("  RRL + GP-ADAPTIVE CBF REAL ROBOT INFERENCE NODE ")
        self.get_logger().info("==================================================")

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

        # CBF & GP Parameters
        self.declare_parameter('cbf_d_min',           0.3)
        self.declare_parameter('cbf_gamma',           3.5)
        self.declare_parameter('cbf_a_max_brake',     2.61)
        self.declare_parameter('cbf_wheelbase',       0.39)
        self.declare_parameter('cbf_fov_deg',         25.0)
        self.declare_parameter('cbf_steer_max',       0.41)
        self.declare_parameter('cbf_slack_weight',    1e4)
        self.declare_parameter('cbf_num_danger_rays', 15)
        self.declare_parameter('enable_gp',           True)
        self.declare_parameter('k_delta',             2.5)
        self.declare_parameter('gp_length_scale',     0.9)
        self.declare_parameter('gp_sigma_f',          0.6)
        self.declare_parameter('gp_sigma_n',          0.03)
        self.declare_parameter('gp_buffer_size',      40)

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

        self.get_logger().info(f"DIL Model : {dil_path}")
        self.get_logger().info(f"RRL Model : {rrl_path}")
        self.get_logger().info(f"Norm JSON : {norm_path}")
        self.get_logger().info(f"Enable RRL: {self.enable_rrl} | Use CBF: {self.use_cbf}")
        self.get_logger().info(f"Drive Topic: {drive_topic} | Odom Topic: {odom_topic} | Scan Topic: {scan_topic}")

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
                self.get_logger().info(f"Loaded norm params from: {norm_path}")
            except Exception as e:
                self.get_logger().warn(f"Failed to read norm JSON ({e}), using embedded default norm params.")
                self.target_mean  = np.array(DEFAULT_MEAN, dtype=np.float32)
                self.target_std   = np.array(DEFAULT_STD, dtype=np.float32)
        else:
            self.target_mean = np.array(DEFAULT_MEAN, dtype=np.float32)
            self.target_std  = np.array(DEFAULT_STD, dtype=np.float32)
            self.get_logger().info("Norm JSON not provided or not found. Used embedded DIL norm parameters.")

        # ── Device ──────────────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Torch device: {self.device}")

        # ── Load DIL Baseline (.pth) ─────────────────────────────────────────
        self.dil_model = DAggerMLP(input_dim=self.target_beams, output_dim=2).to(self.device)
        self.dil_model.load_state_dict(safe_torch_load(dil_path, self.device))
        self.dil_model.eval()
        for p in self.dil_model.parameters():
            p.requires_grad = False
        self.get_logger().info(f"DIL model loaded: {dil_path}")

        # ── Load RRL Policy (.pth) ───────────────────────────────────────────
        self.rrl_model = None
        if self.enable_rrl:
            rrl_state_dim = self.target_beams + 2 + 2 + 2  # 60 + v_x + omega + a_DIL + prev_a_R = 66
            self.rrl_model = RRLActorCritic(rrl_state_dim, action_dim=2).to(self.device)
            self.rrl_model.load_state_dict(safe_torch_load(rrl_path, self.device))
            self.rrl_model.eval()
            self.get_logger().info(f"RRL model loaded: {rrl_path}")

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
            self.get_logger().info(
                f"[GP-CBF INITIALIZED] Active: d_min={self.cbf_d_min}m, gamma={self.cbf_gamma}, "
                f"L={self.cbf_wheelbase}m, a_brake={self.cbf_a_max_brake}m/s², FOV={self.cbf_fov_deg}°, GP={self.enable_gp}"
            )

        # ── State ───────────────────────────────────────────────────────────
        self.latest_scan        = None
        self.latest_scan_angles = None
        self.v_x                = 0.0
        self.v_y                = 0.0
        self.omega              = 0.0
        self.prev_a_R           = np.zeros(2, dtype=np.float32)

        # ── Pub / Sub ───────────────────────────────────────────────────────
        self.pub_drive = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.sub_scan  = self.create_subscription(LaserScan, scan_topic, self._scan_cb, 10)
        self.sub_odom  = self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)

        # Control loop at 20 Hz
        self.create_timer(0.05, self._control_loop)

        self.get_logger().info("Node ready. Waiting for /scan and /odom...")

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self.v_x   = float(msg.twist.twist.linear.x)
        self.v_y   = float(msg.twist.twist.linear.y)
        self.omega = float(msg.twist.twist.angular.z)

    def _scan_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=self.max_range, posinf=self.max_range, neginf=0.0)
        self.latest_scan        = ranges
        self.latest_scan_angles = (
            np.arange(len(ranges), dtype=np.float32) * msg.angle_increment + msg.angle_min
        )

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
        if self.latest_scan is None:
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
        v_total     = float(np.clip(a_dil[0] + a_R[0], self.min_speed, self.max_speed))
        steer_total = float(a_dil[1] + a_R[1])
        u_total     = np.array([v_total, steer_total], dtype=np.float32)

        # 5. CBF Safety Filter
        if self.use_cbf and self.latest_scan_angles is not None:
            u_executed = self.cbf.filter(
                u_total,
                self.latest_scan,
                self.latest_scan_angles,
                v_y_measured=self.v_y
            )
        else:
            u_executed = u_total

        self.prev_a_R = a_R

        # 6. Debug log (throttled every 0.5 s)
        if self.use_cbf and self.enable_gp:
            self.get_logger().info(
                f"DIL: [v={a_dil[0]:.2f}, d={math.degrees(a_dil[1]):+.1f}°] | "
                f"RRL: [Δv={a_R[0]:+.2f}, Δd={math.degrees(a_R[1]):+.1f}°] | "
                f"FINAL: [v={u_executed[0]:.2f}, d={math.degrees(u_executed[1]):+.1f}°] | "
                f"GP: [μ={self.cbf.last_mu_slip:+.3f}, σ={self.cbf.last_sigma_slip:.3f}, m={self.cbf.last_margin_mean:+.2f}]",
                throttle_duration_sec=0.5
            )
        else:
            self.get_logger().info(
                f"DIL: [v={a_dil[0]:.2f}m/s, d={math.degrees(a_dil[1]):+.1f}°] | "
                f"RRL: [Δv={a_R[0]:+.2f}, Δd={math.degrees(a_R[1]):+.1f}°] | "
                f"FINAL: [v={u_executed[0]:.2f}m/s, d={math.degrees(u_executed[1]):+.1f}°]",
                throttle_duration_sec=0.5
            )

        # 7. Publish AckermannDriveStamped
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser'
        msg.drive.speed          = float(u_executed[0])
        msg.drive.steering_angle = float(u_executed[1])
        self.pub_drive.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RRLInferenceRealNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down — sending stop command.")
        stop = AckermannDriveStamped()
        stop.drive.speed = 0.0
        stop.drive.steering_angle = 0.0
        node.pub_drive.publish(stop)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
