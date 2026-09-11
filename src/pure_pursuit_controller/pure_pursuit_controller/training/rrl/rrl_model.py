#!/usr/bin/env python3
"""
rrl_model.py
────────────
PyTorch Neural Network for Residual Reinforcement Learning (RRL/RPL) with PPO.

Features:
- Actor Network: Learns bounded action offsets Δa = [Δv, Δδ].
- Zero-Initialization: Final linear layer of Actor initialized to 0 so at step t=0, Δa ≈ 0.
- Tanh Activation + Scaling: Bounds output within ±1.0 m/s for velocity offset, ±0.05 rad for steering offset.
- Critic Network: Estimates state-value function V(s).
"""

import torch
import torch.nn as nn

class RRLActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 2, scale=(0.4, 0.025)):
        super(RRLActorCritic, self).__init__()
        
        # Scaling factors for action bounds: [max_delta_v (m/s), max_delta_steer (rad)]
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        
        # --- Actor Network (Policy) ---
        self.actor_backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.actor_mean = nn.Linear(256, action_dim)
        
        # Trainable log_std for continuous Gaussian policy PPO
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim) - 0.5)

        # --- Critic Network (Value Function) ---
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # CRUCIAL ZERO-INITIALIZATION TRICK:
        # Initialize actor final output layer weights & bias to ZERO.
        # At start of training, actor outputs 0 -> a_total = a_DIL + 0 = a_DIL.
        nn.init.zeros_(self.actor_mean.weight)
        nn.init.zeros_(self.actor_mean.bias)

    def _map_action(self, mean_raw: torch.Tensor) -> torch.Tensor:
        """Dynamic Asymmetric action mapping for High-Speed & Adaptive Slowdown:
        - Speed offset: [-1.5 m/s, +2.5 m/s]
        - Steering offset: [-0.035 rad, +0.035 rad] (~2.0 deg)
        Preserves zero-init property: mean_raw = 0 -> action_residual = [0, 0]
        Uses torch.where to avoid zero-gradient bottleneck at zero initialization.
        """
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
        """Helper to sample action during rollout or evaluation"""
        action_residual_mean, std, value = self.forward(state)
        if deterministic:
            return action_residual_mean, torch.tensor(0.0), value
        
        dist = torch.distributions.Normal(action_residual_mean, std)
        raw_action = dist.sample()
        # Fix 5: Align clip bounds with _map_action: v in [-1.5, +2.5], steer in [-0.035, +0.035]
        # Previously -0.3 lower bound caused log_prob bias in PPO update (mismatch with _map_action braking range)
        v_clamped = torch.clamp(raw_action[..., 0:1], -1.5, 2.5)
        steer_clamped = torch.clamp(raw_action[..., 1:2], -0.035, 0.035)
        clamped_action = torch.cat([v_clamped, steer_clamped], dim=-1)

        log_prob = dist.log_prob(clamped_action).sum(dim=-1, keepdim=True)
        return clamped_action, log_prob, value

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        """Evaluate log_prob and entropy for PPO policy update batch"""
        features = self.actor_backbone(state)
        mean_raw = self.actor_mean(features)
        action_residual_mean = self._map_action(mean_raw)
        std = torch.exp(self.actor_log_std).expand_as(action_residual_mean)
        
        dist = torch.distributions.Normal(action_residual_mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        value = self.critic(state)
        return log_prob, entropy, value
