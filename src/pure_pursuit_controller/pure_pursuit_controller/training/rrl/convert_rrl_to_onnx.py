#!/usr/bin/env python3
"""
convert_rrl_to_onnx.py
──────────────────────
Converts a trained PyTorch RRL Actor policy (.pth) into ONNX format (.onnx) for ONNXRuntime inference.

Usage:
    python3 -m pure_pursuit_controller.training.rrl.convert_rrl_to_onnx \
        --pth_path /home/adt/f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/models/rrl_ppo_model.pth \
        --onnx_path /home/adt/f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/models/rrl_ppo_model.onnx
"""

import os
import argparse
import torch
import torch.nn as nn
from pure_pursuit_controller.training.rrl.rrl_model import RRLActorCritic


class RRLActorOnly(nn.Module):
    """Wrapper that outputs only the deterministic action offset for ONNX export"""
    def __init__(self, base_model: RRLActorCritic):
        super(RRLActorOnly, self).__init__()
        self.base_model = base_model

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action_residual, _, _ = self.base_model(state)
        return action_residual


def convert_pth_to_onnx(pth_path: str, onnx_path: str, state_dim: int = 66):
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"PyTorch checkpoint file not found: {pth_path}")

    device = torch.device("cpu")
    
    # 1. Load trained PyTorch RRL Actor-Critic model
    rrl_full = RRLActorCritic(state_dim=state_dim, action_dim=2)
    
    # Support SB3 state_dict or raw PyTorch state_dict
    state_dict = torch.load(pth_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    rrl_full.load_state_dict(state_dict, strict=False)
    rrl_full.eval()

    # 2. Extract Actor only wrapper for inference
    actor_export = RRLActorOnly(rrl_full)
    actor_export.eval()

    # 3. Create dummy input tensor matching state dimension (1, 66)
    dummy_input = torch.randn(1, state_dim, dtype=torch.float32)

    # 4. Export to ONNX
    save_dir = os.path.dirname(onnx_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    try:
        torch.onnx.export(
            actor_export,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['state'],
            output_names=['action_residual'],
            dynamic_axes={
                'state': {0: 'batch_size'},
                'action_residual': {0: 'batch_size'}
            },
            dynamo=False
        )
    except TypeError:
        torch.onnx.export(
            actor_export,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['state'],
            output_names=['action_residual'],
            dynamic_axes={
                'state': {0: 'batch_size'},
                'action_residual': {0: 'batch_size'}
            }
        )

    print(f" Successfully converted PyTorch model [{pth_path}] -> ONNX [{onnx_path}]")
    print(f" Input shape:  {dummy_input.shape}")
    print(f" Output shape: {actor_export(dummy_input).shape}")


def main():
    parser = argparse.ArgumentParser(description="Convert RRL PyTorch (.pth) model to ONNX (.onnx)")
    default_pth = "/home/adt/f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/models/rrl_ppo_model.pth"
    default_onnx = "/home/adt/f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/models/rrl_ppo_model.onnx"
    
    parser.add_argument("--pth_path", type=str, default=default_pth, help="Input .pth file path")
    parser.add_argument("--onnx_path", type=str, default=default_onnx, help="Output .onnx file path")
    parser.add_argument("--state_dim", type=int, default=66, help="State dimension")
    
    args = parser.parse_args()
    convert_pth_to_onnx(args.pth_path, args.onnx_path, args.state_dim)


if __name__ == "__main__":
    main()
