#!/usr/bin/env python3
"""
cbf_core.py
───────────
Lớp toán học CBFQPSafetyFilter chịu trách nhiệm lọc an toàn cho xe tự lái F1TENTH.

Chức năng:
  - Nhận u_nominal = [v_nom, delta_nom] từ mô hình AI (Imitation Learning/RL).
  - Đọc dữ liệu khoảng cách và góc của các tia LiDAR.
  - Thiết lập bất đẳng thức Control Barrier Function (CBF) với khoảng cách an toàn d_min.
  - Giải bài toán Quadratic Program (QP) thời gian thực bằng Scipy SLSQP / OSQP.
  - Tích hợp biến nới lỏng (Slack Variable) để tránh Infeasible QP khi tiệm cận tường.
"""

import math
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='qpsolvers')
import numpy as np
import scipy.optimize as opt

try:
    from qpsolvers import solve_qp
    _HAS_QPSOLVERS = True
except ImportError:
    _HAS_QPSOLVERS = False


class CBFQPSafetyFilter:
    def __init__(
        self,
        d_min: float = 0.35,          # Khoảng cách an toàn tối thiểu tới tường/vật cản (m)
        gamma: float = 2.0,           # Hệ số CBF gain gamma
        v_max: float = 3.0,           # Giới hạn vận tốc tối đa (m/s)
        steer_max: float = 0.41,      # Giới hạn góc lái tối đa (rad ~ 23.5 deg)
        slack_weight: float = 1e4,    # Trọng số phạt biến nới lỏng Slack Variable
        num_danger_rays: int = 15,    # Số tia LiDAR nguy hiểm nhất cần đưa vào bài toán QP
        fov_cutoff_deg: float = 10.0  # Góc quét phía trước xét vật cản (+/- độ)
    ):
        self.d_min = d_min
        self.gamma = gamma
        self.v_max = v_max
        self.steer_max = steer_max
        self.slack_weight = slack_weight
        self.num_danger_rays = num_danger_rays
        self.fov_cutoff_rad = math.radians(fov_cutoff_deg)

    def filter(self, u_nominal: np.ndarray, ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """
        Đầu vào:
            u_nominal  : np.array([v_nom, delta_nom])
            ranges     : mảng khoảng cách đo được từ LiDAR (m)
            angles     : mảng góc tương ứng của từng tia LiDAR (rad)

        Đầu ra:
            u_safe     : np.array([v_safe, delta_safe])
        """
        v_nom = float(u_nominal[0])
        delta_nom = float(u_nominal[1])

        # 1. Trích xuất các ràng buộc CBF từ LiDAR
        G_cbf, h_cbf = self._extract_cbf_constraints(ranges, angles)

        # Nếu phía trước không có vật cản gần -> Giữ nguyên u_nominal
        if G_cbf is None or len(G_cbf) == 0:
            return np.array([v_nom, delta_nom], dtype=np.float32)

        # 2. Thử giải QP bằng qpsolvers nếu có solver backend (OSQP)
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

        # 3. Giải QP tối ưu bằng SciPy SLSQP (Luôn sẵn có, chạy rất nhanh < 1ms cho 3 biến)
        return self._solve_scipy_qp(v_nom, delta_nom, G_cbf, h_cbf)

    def _solve_scipy_qp(self, v_nom: float, delta_nom: float, G: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Giải bài toán QP 3 biến [v, delta, slack] bằng SciPy SLSQP"""
        def objective(x):
            v, steer, slack = x[0], x[1], x[2]
            return 0.5 * (v - v_nom)**2 + 2.5 * (steer - delta_nom)**2 + 0.5 * self.slack_weight * (slack**2)

        def jacobian(x):
            v, steer, slack = x[0], x[1], x[2]
            return np.array([v - v_nom, 5.0 * (steer - delta_nom), self.slack_weight * slack])

        # Constraint: h - G * x >= 0
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

        # Fallback khẩn cấp nếu SciPy không hội tụ: Cho v_safe chạy chậm vừa đủ (0.3 m/s) để xe không chết dí
        return np.array([max(0.2, min(v_nom * 0.3, self.v_max)), delta_nom], dtype=np.float32)

    def _extract_cbf_constraints(self, ranges: np.ndarray, angles: np.ndarray):
        """Tạo ma trận G và vector h cho bất đẳng thức G * x <= h từ dữ liệu LiDAR"""
        mask_front = (angles >= -self.fov_cutoff_rad) & (angles <= self.fov_cutoff_rad)
        valid_ranges = ranges[mask_front]
        valid_angles = angles[mask_front]

        valid_mask = ~np.isnan(valid_ranges) & ~np.isinf(valid_ranges) & (valid_ranges > 0.01)
        if not np.any(valid_mask):
            return None, None

        valid_ranges = valid_ranges[valid_mask]
        valid_angles = valid_angles[valid_mask]

        # Chọn k tia có khoảng cách đo nhỏ nhất (vật cản/tường sát xe nhất)
        danger_indices = np.argsort(valid_ranges)[:self.num_danger_rays]

        G_list = []
        h_list = []

        for idx in danger_indices:
            r_i = float(valid_ranges[idx])
            phi_i = float(valid_angles[idx])

            # Hàm Barrier: h_i = r_i - d_min >= 0
            h_val = r_i - self.d_min

            # Đạo hàm: dh_i/dt = -v * cos(phi_i)
            # CBF Constraint: -v * cos(phi_i) + gamma * h_val >= -slack
            # <=> v * cos(phi_i) - slack <= gamma * (r_i - d_min)
            G_list.append([math.cos(phi_i), 0.0, -1.0])
            h_list.append(self.gamma * h_val)

        return np.array(G_list), np.array(h_list)
