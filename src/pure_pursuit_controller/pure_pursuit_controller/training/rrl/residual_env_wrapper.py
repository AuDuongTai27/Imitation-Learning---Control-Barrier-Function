#!/usr/bin/env python3
"""
residual_env_wrapper.py
───────────────────────
OpenAI Gym Environment Wrapper for Residual Reinforcement Learning (RRL) in F1TENTH.

Highlights:
1. Loads frozen PyTorch DAggerMLP baseline policy.
2. Downsamples & normalizes raw 1080 LiDAR scan to 60 beams (matching DAggerMLP training data).
3. Aggregates RRL state observation: [LiDAR_60, v_x, omega, v_DIL, delta_DIL, prev_Δv, prev_Δδ].
4. Computes total control: a_total = a_DIL(s) + a_R(s).
5. Optional CBF-QP Safety Filter execution inside step().
6. Reward Function: r = τ1 * v_x + τ2 * v_y^2 + ρ * I(collision).
"""

import os
import json
import math
import gym
import numpy as np
import torch

from pure_pursuit_controller.training.train import DAggerMLP
from pure_pursuit_controller.cbf.cbf_core import CBFQPSafetyFilter


class F1TenthResidualEnvWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        dagger_model_path: str,
        norm_param_path: str,
        use_cbf: bool = True,
        tau1: float = 1.0,
        tau2: float = -0.005,
        rho: float = -10.0,
        scale=(0.4, 0.025)
    ):
        super(F1TenthResidualEnvWrapper, self).__init__(env)
        self.env = env
        self.use_cbf = use_cbf
        self.tau1 = tau1
        self.tau2 = tau2
        self.rho = rho
        self.scale = np.array(scale, dtype=np.float32)

        # 1. Load Normalization & DAgger DIL Baseline Model
        if not os.path.exists(norm_param_path):
            raise FileNotFoundError(f"Normalization file not found: {norm_param_path}")
        if not os.path.exists(dagger_model_path):
            raise FileNotFoundError(f"DAgger model file not found: {dagger_model_path}")

        with open(norm_param_path, 'r') as f:
            norm_data = json.load(f)
        self.target_beams = norm_data.get("target_beams", 60)
        self.max_range = norm_data.get("max_range", 10.0)

        if "target_mean" in norm_data and "target_std" in norm_data:
            self.target_mean = np.array(norm_data["target_mean"], dtype=np.float32)
            self.target_std = np.array(norm_data["target_std"], dtype=np.float32)
        else:
            self.target_mean = None
            self.target_std = None

        self.dil_is_onnx = dagger_model_path.endswith('.onnx')
        if self.dil_is_onnx:
            import onnxruntime as ort
            self.dil_ort_session = ort.InferenceSession(dagger_model_path)
            self.dil_input_name = self.dil_ort_session.get_inputs()[0].name
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dil_model = DAggerMLP(input_dim=self.target_beams, output_dim=2).to(self.device)
            self.dil_model.load_state_dict(torch.load(dagger_model_path, map_location=self.device))
            self.dil_model.eval()
            for p in self.dil_model.parameters():
                p.requires_grad = False

        # 2. Instantiate CBF Safety Filter
        if self.use_cbf:
            self.cbf = CBFQPSafetyFilter(d_min=0.30, gamma=1.5, v_max=5.0, steer_max=0.41)

        # 3. Observation Space: 60 beams + v_x + omega + a_DIL (2) + prev_a_R (2) = 66 dimensions
        self.obs_dim = self.target_beams + 2 + 2 + 2
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # Action Space for RRL Policy (Normalized [-1, 1] offset)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.prev_a_R = np.zeros(2, dtype=np.float32)

    def _preprocess_scan(self, raw_scan: np.ndarray):
        """Crop raw LiDAR scan to frontal [-60, 60] degrees FOV and resample to 60 beams"""
        angles = np.linspace(-math.pi * 3/4, math.pi * 3/4, len(raw_scan))
        crop_limit = math.radians(60.0)
        mask = (angles >= -crop_limit) & (angles <= crop_limit)

        if not np.any(mask):
            return np.ones(self.target_beams, dtype=np.float32) * self.max_range, np.ones(self.target_beams, dtype=np.float32)

        valid_ranges = np.clip(raw_scan[mask], 0.0, self.max_range)
        valid_angles = angles[mask]

        target_angles = np.linspace(-crop_limit, crop_limit, self.target_beams)
        interpolated = np.interp(target_angles, valid_angles, valid_ranges)
        scan_meters = interpolated.astype(np.float32)
        scan_norm = (interpolated / self.max_range).astype(np.float32)
        return scan_meters, scan_norm

    def reset(self, poses=None, **kwargs):
        if poses is None:
            num_agents = getattr(self.env.unwrapped, 'num_agents', 1)
            poses = np.zeros((num_agents, 3), dtype=np.float32)
        try:
            res = self.env.reset(poses=poses, **kwargs)
        except (TypeError, ValueError):
            res = self.env.reset(poses, **kwargs)

        if isinstance(res, tuple):
            gym_obs = res[0]
            info = res[-1] if isinstance(res[-1], dict) else {}
        else:
            gym_obs = res
            info = {}

        self.prev_a_R = np.zeros(2, dtype=np.float32)
        self.last_gym_obs = gym_obs
        rrl_obs, _ = self._extract_state(gym_obs)
        return rrl_obs, info

    def _extract_state(self, gym_obs):
        """Extract LiDAR and vehicle state parameters from F1TENTH Gym observation dict"""
        if isinstance(gym_obs, dict):
            raw_scan = gym_obs['scans'][0] if 'scans' in gym_obs else gym_obs.get('scan', np.zeros(1080))
            v_x = float(gym_obs.get('linear_vels_x', [0.0])[0]) if 'linear_vels_x' in gym_obs else 0.0
            omega = float(gym_obs.get('ang_vels_z', [0.0])[0]) if 'ang_vels_z' in gym_obs else 0.0
        else:
            raw_scan = gym_obs[:1080]
            v_x = float(gym_obs[1080]) if len(gym_obs) > 1080 else 0.0
            omega = float(gym_obs[1081]) if len(gym_obs) > 1081 else 0.0

        scan_meters, scan_norm = self._preprocess_scan(raw_scan)

        # Run DAgger DIL Model Inference
        if self.dil_is_onnx:
            if self.dil_input_name == 'lidar_raw':
                tensor_input = scan_meters.reshape(1, -1).astype(np.float32)
                outputs = self.dil_ort_session.run(None, {self.dil_input_name: tensor_input})
                a_dil = outputs[0].squeeze(0).astype(np.float32)
            else:
                tensor_input = scan_norm.reshape(1, -1).astype(np.float32)
                outputs = self.dil_ort_session.run(None, {self.dil_input_name: tensor_input})
                a_dil = outputs[0].squeeze(0).astype(np.float32)
                if self.target_mean is not None and self.target_std is not None:
                    a_dil = a_dil * self.target_std + self.target_mean
        else:
            scan_tensor = torch.tensor(scan_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                a_dil = self.dil_model(scan_tensor).cpu().numpy().squeeze(0)
            if self.target_mean is not None and self.target_std is not None:
                a_dil = a_dil * self.target_std + self.target_mean

        rrl_obs = np.hstack([scan_norm, [v_x, omega], a_dil, self.prev_a_R]).astype(np.float32)
        return rrl_obs, a_dil

    def step(self, action_rrl_norm: np.ndarray):
        # Scale RRL policy output [-1, 1] to actual offsets [±1.0 m/s, ±0.05 rad]
        a_R = action_rrl_norm * self.scale

        # Get current state and nominal DIL action from last observation
        gym_raw_obs = self.last_gym_obs
        rrl_obs, a_dil = self._extract_state(gym_raw_obs)

        # Total action = DIL baseline + RRL corrective offset
        v_total = float(a_dil[0] + a_R[0])
        steer_total = float(a_dil[1] + a_R[1])
        u_total = np.array([v_total, steer_total], dtype=np.float32)

        # Filter through CBF-QP Safety Layer if enabled
        if self.use_cbf:
            raw_scan = gym_raw_obs['scans'][0] if isinstance(gym_raw_obs, dict) and 'scans' in gym_raw_obs else np.ones(1080) * 8.0
            angles = np.linspace(-math.pi * 3/4, math.pi * 3/4, len(raw_scan))
            u_executed = self.cbf.filter(u_total, raw_scan, angles)
        else:
            u_executed = u_total

        # Execute control in F1TENTH environment (Format expected: [[steering, speed]])
        num_agents = getattr(self.env.unwrapped, 'num_agents', 1)
        action_matrix = np.zeros((num_agents, 2), dtype=np.float32)
        action_matrix[0] = [u_executed[1], u_executed[0]]
        step_res = self.env.step(action_matrix)
        if len(step_res) == 5:
            obs, reward_raw, done, truncated, info = step_res
        else:
            obs, reward_raw, done, info = step_res
            truncated = False

        # Extract vehicle velocity components for reward calculation
        if isinstance(obs, dict):
            v_x = float(obs.get('linear_vels_x', [0.0])[0])
            v_y = float(obs.get('linear_vels_y', [0.0])[0])
            collision = bool(info.get('collisions', [0])[0])
        else:
            v_x, v_y, collision = 0.0, 0.0, bool(done)

        # Step Reward Calculation:
        # 1. Forward Speed progress: tau1 * v_x
        # 2. Penalty for lateral velocity (drifting/skidding): tau2 * v_y^2
        # 3. Action Smoothness Penalty: penalize change in residual actions to avoid high-frequency chatter
        delta_action = a_R - self.prev_a_R
        action_smoothness_penalty = 0.5 * np.sum(delta_action ** 2) + 0.1 * np.sum(a_R ** 2)
        
        # 4. Crawling / Stopping Penalty: penalize v_x < 1.0 m/s to prevent conservative stopping trap
        stopping_penalty = 0.0
        if v_x < 1.0:
            stopping_penalty = (1.0 - v_x) * 2.0

        rrl_reward = (self.tau1 * v_x) + (self.tau2 * (v_y ** 2)) - action_smoothness_penalty - stopping_penalty
        if collision:
            rrl_reward += self.rho

        self.prev_a_R = a_R
        self.last_gym_obs = obs
        next_rrl_obs, _ = self._extract_state(obs)

        return next_rrl_obs, float(rrl_reward), done, truncated, info
