# SAFE SIM-TO-REAL TRANSFER FOR END-TO-END AUTONOMOUS RACING VIA IMITATION LEARNING AND CONTROL BARRIER FUNCTION SAFETY FILTERING

Official codebase for **EIUSC 2026**.

## Authors
- **Au Duong Tai**
- **Huynh Cong Danh**
- **Tran Duy Nhat**

---

## Overview

This repository presents a safe Sim-to-Real transfer framework for end-to-end autonomous racing on 1/10th scale F1TENTH vehicles. The system combines:
1. **End-to-End Imitation Learning (IL)**: Deep neural network policies mapping 2D LiDAR scans directly to Ackermann control commands.
2. **Offline Iterative DAgger**: Data aggregation mechanism using expert path planning (Pure Pursuit / RRT) to mitigate covariate shift during execution.
3. **Control Barrier Function (CBF-QP) Safety Filter**: Real-time Quadratic Programming safety filter enforcing forward invariance of safe state sets, guaranteeing collision avoidance without compromising high-speed performance.

---

## Repository Structure

```text
f1_ws/
├── benchmark_results/        # Logged frequency and latency benchmark CSVs/plots
└── src/
    └── pure_pursuit_controller/
        ├── CMakeLists.txt
        ├── package.xml
        └── pure_pursuit_controller/
            ├── ai_inference/       # AI inference nodes (ONNX / PyTorch JIT for Sim & Real)
            ├── cbf/                # Control Barrier Function (CBF-QP) safety filter core & node
            ├── controllers/        # Control baselines (Pure Pursuit, MPPI)
            ├── data_collection/    # Dataset recording utilities
            ├── models/             # Pre-trained models and normalization stats (.pth, .onnx, _norm.json)
            ├── rrt_planner/        # Expert RRT* path planning nodes
            ├── tools/              # Real-time frequency benchmark & visualization tools
            └── training/           # Offline training notebooks & dataset preprocessing
```

---

## Prerequisites & Installation

### Environment Requirements
- ROS 2 (Humble / Foxy)
- Python 3.10+
- PyTorch / ONNX Runtime
- SciPy / `qpsolvers` (OSQP backend)

### Build Instructions

```bash
cd ~/f1_ws
colcon build --packages-select pure_pursuit_controller
source install/setup.bash
```

---

## Usage

### 1. AI Inference (Real Vehicle PyTorch)
```bash
ros2 run pure_pursuit_controller ai_inference_real_pytorch
```

### 2. CBF Safety Filter Node
```bash
ros2 run pure_pursuit_controller cbf_safety_filter_node
```

### 3. Real-Time Frequency Benchmark
```bash
ros2 run pure_pursuit_controller benchmark_frequency
python3 src/pure_pursuit_controller/pure_pursuit_controller/tools/visualize_benchmark.py
```

---

## License

Developed for the **EIUSC 2026** competition. All rights reserved.
