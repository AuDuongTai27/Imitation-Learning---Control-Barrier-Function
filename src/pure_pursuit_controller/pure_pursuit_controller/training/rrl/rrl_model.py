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

    def forward(self, state: torch.Tensor):
        features = self.actor_backbone(state)
        # Tanh outputs in range [-1, 1], then multiplied by scale vector
        mean_normalized = torch.tanh(self.actor_mean(features))
        action_residual = mean_normalized * self.scale
        
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
        # Clamp to bounds specified by scale
        clamped_action = torch.clamp(raw_action, -self.scale, self.scale)
        log_prob = dist.log_prob(clamped_action).sum(dim=-1, keepdim=True)
        return clamped_action, log_prob, value

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        """Evaluate log_prob and entropy for PPO policy update batch"""
        features = self.actor_backbone(state)
        mean_normalized = torch.tanh(self.actor_mean(features))
        action_residual_mean = mean_normalized * self.scale
        std = torch.exp(self.actor_log_std).expand_as(action_residual_mean)
        
        dist = torch.distributions.Normal(action_residual_mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        value = self.critic(state)
        return log_prob, entropy, value
