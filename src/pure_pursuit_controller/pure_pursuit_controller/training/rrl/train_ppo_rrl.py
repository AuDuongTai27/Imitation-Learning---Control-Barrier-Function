#!/usr/bin/env python3
"""
train_ppo_rrl.py
────────────────
Local Training Script for Residual Reinforcement Learning (RRL) using PPO on F1TENTH.

Features:
- Fast Local Training (Native F1TENTH Gym API > 1000 FPS).
- Integrates Frozen DAgger Baseline + CBF-QP Safety Layer.
- Complete PPO Gradient Updates (GAE Advantage, Clipped Policy Loss, Value Loss).
- Supports Stable-Baselines3 (SB3) or Custom PyTorch PPO update loop.
- Automatic ONNX export upon training completion.
"""

import os
import sys
import time
import argparse
import gym
import numpy as np
import torch
import torch.nn as nn

from pure_pursuit_controller.training.rrl.residual_env_wrapper import F1TenthResidualEnvWrapper
from pure_pursuit_controller.training.rrl.rrl_model import RRLActorCritic

# Try importing Stable-Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


def train_custom_ppo(env, args):
    """Custom PyTorch PPO training loop with full GAE advantages & gradient updates"""
    print(">>> Running Custom PyTorch PPO Training Loop with Policy Gradient Updates...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    model = RRLActorCritic(state_dim, action_dim, scale=(1.0, 0.05)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    save_dir = os.path.dirname(args.output_model)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    obs, _ = env.reset()
    total_steps = 0
    num_episodes = 0
    episode_reward = 0.0
    start_time = time.time()

    rollout_states = []
    rollout_actions = []
    rollout_log_probs = []
    rollout_rewards = []
    rollout_values = []
    rollout_dones = []

    print(f">>> Target timesteps: {args.total_timesteps}. Rollout horizon: {args.rollout_horizon}. Batch size: {args.batch_size}")

    while total_steps < args.total_timesteps:
        # 1. Collect Rollouts of length args.rollout_horizon
        for _ in range(args.rollout_horizon):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action_rrl, log_prob, val = model.get_action(obs_tensor)

            action_rrl_np = action_rrl.cpu().numpy().squeeze(0)
            next_obs, reward, done, truncated, info = env.step(action_rrl_np)
            
            rollout_states.append(obs)
            rollout_actions.append(action_rrl_np)
            rollout_log_probs.append(log_prob.cpu().item())
            rollout_rewards.append(reward)
            rollout_values.append(val.cpu().item())
            rollout_dones.append(float(done or truncated))

            episode_reward += reward
            total_steps += 1
            obs = next_obs

            if done or truncated:
                num_episodes += 1
                obs, _ = env.reset()
                episode_reward = 0.0

            if total_steps >= args.total_timesteps:
                break

        # 2. Compute Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            last_obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            _, _, next_val = model(last_obs_tensor)
            next_value = next_val.cpu().item()

        returns = []
        advantages = []
        gae = 0.0
        
        for t in reversed(range(len(rollout_rewards))):
            next_v = next_value if t == len(rollout_rewards) - 1 else rollout_values[t + 1]
            non_terminal = 1.0 - rollout_dones[t]
            delta = rollout_rewards[t] + args.gamma * next_v * non_terminal - rollout_values[t]
            gae = delta + args.gamma * 0.95 * non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + rollout_values[t])

        b_states = torch.tensor(np.array(rollout_states), dtype=torch.float32).to(device)
        b_actions = torch.tensor(np.array(rollout_actions), dtype=torch.float32).to(device)
        b_log_probs = torch.tensor(np.array(rollout_log_probs), dtype=torch.float32).unsqueeze(-1).to(device)
        b_advantages = torch.tensor(np.array(advantages), dtype=torch.float32).unsqueeze(-1).to(device)
        b_returns = torch.tensor(np.array(returns), dtype=torch.float32).unsqueeze(-1).to(device)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # 3. PPO Update Epochs
        for _ in range(4):  # 4 PPO epochs
            indices = np.arange(len(rollout_states))
            np.random.shuffle(indices)
            
            for start_idx in range(0, len(rollout_states), args.batch_size):
                batch_idx = indices[start_idx:start_idx + args.batch_size]
                if len(batch_idx) < 32:
                    continue

                mb_states = b_states[batch_idx]
                mb_actions = b_actions[batch_idx]
                mb_log_probs = b_log_probs[batch_idx]
                mb_advantages = b_advantages[batch_idx]
                mb_returns = b_returns[batch_idx]

                new_log_probs, entropy, new_values = model.evaluate_actions(mb_states, mb_actions)

                # Ratio: π_new / π_old
                ratio = torch.exp(new_log_probs - mb_log_probs)

                # Clipped Surrogate Objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value Function Loss
                critic_loss = 0.5 * nn.MSELoss()(new_values, mb_returns)

                # Total PPO Loss
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

        # Clear Rollout Buffer
        rollout_states.clear()
        rollout_actions.clear()
        rollout_log_probs.clear()
        rollout_rewards.clear()
        rollout_values.clear()
        rollout_dones.clear()

        # Progress Report
        elapsed = time.time() - start_time
        fps = total_steps / elapsed if elapsed > 0 else 0
        eta = (args.total_timesteps - total_steps) / fps if fps > 0 else 0
        pct = (total_steps / args.total_timesteps) * 100
        print(f"--> [PROGRESS {pct:.1f}%] Step {total_steps}/{args.total_timesteps} | Speed: {fps:.0f} FPS | Elapsed: {elapsed:.1f}s | Remaining: ~{eta:.1f}s")

    torch.save(model.state_dict(), args.output_model)
    print(f">>> Custom PPO Training Finished! Model saved to {args.output_model}")

    # Auto convert to ONNX format
    try:
        from pure_pursuit_controller.training.rrl.convert_rrl_to_onnx import convert_pth_to_onnx
        onnx_output = args.output_model.replace('.pth', '.onnx')
        convert_pth_to_onnx(args.output_model, onnx_output, state_dim=state_dim)
    except Exception as e:
        print(f"[Warning] Automatic ONNX conversion skipped: {e}")


def train_sb3_ppo(raw_env, dagger_model_path, norm_param_path, args):
    """Train RRL policy using Stable-Baselines3 PPO"""
    print(">>> Running Stable-Baselines3 PPO Training Loop...")
    
    def make_env():
        return F1TenthResidualEnvWrapper(
            raw_env,
            dagger_model_path=dagger_model_path,
            norm_param_path=norm_param_path,
            use_cbf=args.use_cbf,
            scale=(1.0, 0.05)
        )

    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.rollout_horizon,
        batch_size=args.batch_size,
        gamma=args.gamma,
        clip_range=args.clip_eps,
        target_kl=args.target_kl,
        verbose=1,
        tensorboard_log=args.tensorboard_dir
    )

    print(f">>> Training PPO for {args.total_timesteps} timesteps...")
    model.learn(total_timesteps=args.total_timesteps)

    model.save(args.output_model.replace('.pth', ''))
    print(f">>> SB3 Training Complete! Saved model to {args.output_model}")

    try:
        from pure_pursuit_controller.training.rrl.convert_rrl_to_onnx import convert_pth_to_onnx
        onnx_output = args.output_model.replace('.pth', '.onnx')
        convert_pth_to_onnx(args.output_model, onnx_output, state_dim=66)
    except Exception as e:
        print(f"[Warning] Automatic ONNX conversion skipped: {e}")


def main():
    parser = argparse.ArgumentParser(description="Local RRL-PPO Training Script for F1TENTH")
    default_map = "/home/adt/f1_ws/src/f1tenth_gym_ros/maps/Spielberg_map"
    models_dir = "/home/adt/f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/models"
    default_dagger = os.path.join(models_dir, "final_combined_37500.pth")
    if not os.path.exists(default_dagger):
        default_dagger = os.path.join(models_dir, "final_combined_37500.onnx")
    default_norm = os.path.join(models_dir, "final_combined_37500_norm.json")
    default_output = os.path.join(models_dir, "rrl_ppo_model.pth")

    parser.add_argument("--map", type=str, default=default_map, help="Map file path or map name in f1tenth_gym_ros/maps")
    parser.add_argument("--dagger_model", type=str, default=default_dagger)
    parser.add_argument("--norm_param", type=str, default=default_norm)
    parser.add_argument("--output_model", type=str, default=default_output)
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Total RL training steps")
    parser.add_argument("--rollout_horizon", type=int, default=2048, help="PPO rollout steps")
    parser.add_argument("--batch_size", type=int, default=128, help="PPO minibatch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.998)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--target_kl", type=float, default=0.01)
    parser.add_argument("--use_cbf", action="store_true", default=True, help="Enable CBF Safety Filter in loop")
    parser.add_argument("--tensorboard_dir", type=str, default="./tensorboard_rrl_logs/")

    args = parser.parse_args()

    # Resolve map path if a short map name was provided
    map_path = args.map
    if not os.path.exists(map_path + ".yaml") and not os.path.exists(map_path + ".png"):
        candidate_path = os.path.join("/home/adt/f1_ws/src/f1tenth_gym_ros/maps", args.map)
        if os.path.exists(candidate_path + ".yaml") or os.path.exists(candidate_path + ".png"):
            map_path = candidate_path

    # Determine map extension (.png or .pgm)
    map_ext = ".png"
    if os.path.exists(map_path + ".pgm"):
        map_ext = ".pgm"

    # Dynamic Gym import (supports f110_gym or f1tenth_gym)
    raw_env = None
    try:
        import f110_gym
        raw_env = gym.make('f110_gym:f110-v0', map=map_path, map_ext=map_ext, num_agents=1, timestep=0.01)
    except Exception as e1:
        try:
            import f1tenth_gym
            raw_env = gym.make('f1tenth_gym:f1tenth-v0', map=map_path, num_agents=1, timestep=0.01)
        except Exception as e2:
            print(f"[Error] Could not initialize F1TENTH Gym (f110_gym error: {e1}, f1tenth_gym error: {e2}).")
            sys.exit(1)

    if HAS_SB3:
        train_sb3_ppo(raw_env, args.dagger_model, args.norm_param, args)
    else:
        env = F1TenthResidualEnvWrapper(
            raw_env,
            dagger_model_path=args.dagger_model,
            norm_param_path=args.norm_param,
            use_cbf=args.use_cbf
        )
        train_custom_ppo(env, args)


if __name__ == "__main__":
    main()
