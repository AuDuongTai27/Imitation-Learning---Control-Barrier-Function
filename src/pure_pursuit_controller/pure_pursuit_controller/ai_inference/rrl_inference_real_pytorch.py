#!/usr/bin/env python3
"""
rrl_inference_real_pytorch.py
─────────────────────────────
ROS 2 Node: Full RRL Pipeline for REAL F1TENTH Robot (PyTorch .pth files).

Pipeline:
  /scan  ──► Preprocess (60 beams, norm) ──► DIL (.pth) ──► a_DIL = [v, δ]
                                         ──► RRL State (66-dim)
                                         ──► RRL (.pth) ──► a_R  = [Δv, Δδ]
                                         ──► a_total = a_DIL + a_R
                                         ──► CBF-QP Filter ──► /drive

Parameters (set via --ros-args -p):
  dil_model_path   : Path to DIL .pth model  (required)
  rrl_model_path   : Path to RRL .pth model  (required)
  norm_param_path  : Path to _norm.json      (required for DIL .pth)
  enable_rrl       : bool (default: True)  — set False for Pure DIL mode
  use_cbf          : bool (default: True)  — set False to bypass CBF
  drive_topic      : string (default: /drive)
  max_speed        : float (default: 7.0)  — absolute max speed (m/s)
  min_speed        : float (default: 0.0)  — allow full stop (real robot)

Diagnostic Modes:
  Pure DIL only:      enable_rrl:=false use_cbf:=false
  DIL + CBF:          enable_rrl:=false use_cbf:=true
  Full (DIL+RRL+CBF): enable_rrl:=true  use_cbf:=true
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
    from qpsolvers import solve_qp
    _HAS_QPSOLVERS = True
except ImportError:
    _HAS_QPSOLVERS = False


class CBFQPSafetyFilter:
    def __init__(
        self,
        d_min: float = 0.35,
        gamma: float = 2.0,
        v_max: float = 3.0,
        steer_max: float = 0.41,
        slack_weight: float = 1e4,
        num_danger_rays: int = 15,
        fov_cutoff_deg: float = 15.0
    ):
        self.d_min = d_min
        self.gamma = gamma
        self.v_max = v_max
        self.steer_max = steer_max
        self.slack_weight = slack_weight        
        self.num_danger_rays = num_danger_rays
        self.fov_cutoff_rad = math.radians(fov_cutoff_deg)

    def filter(self, u_nominal: np.ndarray, ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
        v_nom = float(u_nominal[0])
        delta_nom = float(u_nominal[1])

        G_cbf, h_cbf = self._extract_cbf_constraints(ranges, angles)

        if G_cbf is None or len(G_cbf) == 0:
            return np.array([v_nom, delta_nom], dtype=np.float32)

        if _HAS_QPSOLVERS:
            try:
                P = np.diag([1.0, 5.0, self.slack_weight]).astype(np.float64)
                q = np.array([-v_nom, -5.0 * delta_nom, 0.0], dtype=np.float64)
                lb = np.array([0.0, -self.steer_max, 0.0], dtype=np.float64)
                ub = np.array([self.v_max, self.steer_max, 10.0], dtype=np.float64)
                sol = solve_qp(P, q, G_cbf, h_cbf, None, None, lb, ub, solver="osqp")
                if sol is not None:
                    v_safe = float(np.clip(sol[0], 0.0, self.v_max))
                    delta_safe = float(np.clip(sol[1], -self.steer_max, self.steer_max))
                    return np.array([v_safe, delta_safe], dtype=np.float32)
            except Exception:
                pass

        return self._solve_scipy_qp(v_nom, delta_nom, G_cbf, h_cbf)

    def _solve_scipy_qp(self, v_nom: float, delta_nom: float, G: np.ndarray, h: np.ndarray) -> np.ndarray:
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

        x0 = np.array([v_nom, delta_nom, 0.0])
        res = opt.minimize(objective, x0, method='SLSQP', jac=jacobian, constraints=constraints, bounds=bounds)

        if res.success and res.x is not None:
            v_safe = float(np.clip(res.x[0], 0.0, self.v_max))
            delta_safe = float(np.clip(res.x[1], -self.steer_max, self.steer_max))
            return np.array([v_safe, delta_safe], dtype=np.float32)

        return np.array([max(0.2, min(v_nom * 0.3, self.v_max)), delta_nom], dtype=np.float32)

    def _extract_cbf_constraints(self, ranges: np.ndarray, angles: np.ndarray):
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

        for idx in danger_indices:
            r_i = float(valid_ranges[idx])
            phi_i = float(valid_angles[idx])

            h_val = r_i - self.d_min
            G_list.append([math.cos(phi_i), 0.0, -1.0])
            h_list.append(self.gamma * h_val)

        return np.array(G_list), np.array(h_list)


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


class RRLInferenceRealNode(Node):
    def __init__(self):
        super().__init__('rrl_inference_real_node')

        self.get_logger().info("=========================================")
        self.get_logger().info("  RRL REAL ROBOT INFERENCE NODE STARTED  ")
        self.get_logger().info("=========================================")

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('dil_model_path',  'final_combined_37500.pth')
        self.declare_parameter('rrl_model_path',  'rrl_ppo_model.pth')
        self.declare_parameter('norm_param_path', 'final_combined_37500_norm.json')
        self.declare_parameter('enable_rrl',   True)
        self.declare_parameter('use_cbf',      True)
        self.declare_parameter('cbf_fov_deg',  15.0)
        self.declare_parameter('drive_topic',  '/drive')
        self.declare_parameter('odom_topic',   '/odom')
        self.declare_parameter('max_speed',    7.0)
        self.declare_parameter('min_speed',    0.0)

        dil_path   = resolve_model_path(self.get_parameter('dil_model_path').get_parameter_value().string_value)
        rrl_path   = resolve_model_path(self.get_parameter('rrl_model_path').get_parameter_value().string_value)
        norm_path  = resolve_model_path(self.get_parameter('norm_param_path').get_parameter_value().string_value)
        self.enable_rrl   = self.get_parameter('enable_rrl').get_parameter_value().bool_value
        self.use_cbf      = self.get_parameter('use_cbf').get_parameter_value().bool_value
        self.cbf_fov_deg  = self.get_parameter('cbf_fov_deg').get_parameter_value().double_value
        drive_topic       = self.get_parameter('drive_topic').get_parameter_value().string_value
        odom_topic        = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.max_speed    = self.get_parameter('max_speed').get_parameter_value().double_value
        self.min_speed    = self.get_parameter('min_speed').get_parameter_value().double_value

        self.get_logger().info(f"DIL Model : {dil_path}")
        self.get_logger().info(f"RRL Model : {rrl_path}")
        self.get_logger().info(f"Norm JSON : {norm_path}")
        self.get_logger().info(f"Enable RRL: {self.enable_rrl} | Use CBF: {self.use_cbf}")
        self.get_logger().info(f"Drive Topic: {drive_topic} | Odom Topic: {odom_topic}")

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
        self.dil_model.load_state_dict(torch.load(dil_path, map_location=self.device))
        self.dil_model.eval()
        for p in self.dil_model.parameters():
            p.requires_grad = False
        self.get_logger().info(f"DIL model loaded: {dil_path}")

        # ── Load RRL Policy (.pth) ───────────────────────────────────────────
        self.rrl_model = None
        if self.enable_rrl:
            rrl_state_dim = self.target_beams + 2 + 2 + 2  # 60 + v_x + omega + a_DIL + prev_a_R = 66
            self.rrl_model = RRLActorCritic(rrl_state_dim, action_dim=2).to(self.device)
            self.rrl_model.load_state_dict(torch.load(rrl_path, map_location=self.device))
            self.rrl_model.eval()
            self.get_logger().info(f"RRL model loaded: {rrl_path}")

        # ── CBF Safety Filter ───────────────────────────────────────────────
        if self.use_cbf:
            self.cbf = CBFQPSafetyFilter(
                d_min=0.30, gamma=1.5, v_max=self.max_speed, steer_max=0.41, fov_cutoff_deg=self.cbf_fov_deg
            )
            self.get_logger().info(f"CBF-QP safety filter initialized (FOV: +/-{self.cbf_fov_deg} deg).")

        # ── State ───────────────────────────────────────────────────────────
        self.latest_scan        = None
        self.latest_scan_angles = None
        self.v_x                = 0.0
        self.omega              = 0.0
        self.prev_a_R           = np.zeros(2, dtype=np.float32)

        # ── Pub / Sub ───────────────────────────────────────────────────────
        self.pub_drive = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.sub_scan  = self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.sub_odom  = self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)

        # Control loop at 20 Hz
        self.create_timer(0.05, self._control_loop)

        self.get_logger().info("Node ready. Waiting for /scan and /odom...")

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self.v_x   = float(msg.twist.twist.linear.x)
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
            u_executed = self.cbf.filter(u_total, self.latest_scan, self.latest_scan_angles)
        else:
            u_executed = u_total

        self.prev_a_R = a_R

        # 6. Debug log (throttled every 0.5 s)
        self.get_logger().info(
            f"DIL: [v={a_dil[0]:.2f}m/s, d={math.degrees(a_dil[1]):+.1f}] | "
            f"RRL: [Dv={a_R[0]:+.2f}, Dd={math.degrees(a_R[1]):+.1f}] | "
            f"FINAL: [v={u_executed[0]:.2f}m/s, d={math.degrees(u_executed[1]):+.1f}]",
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
