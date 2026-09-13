#!/usr/bin/env python3
"""
cbf_core.py
───────────
Lớp toán học CBFQPSafetyFilter chịu trách nhiệm lọc an toàn cho xe tự lái F1TENTH,
đã được NÂNG CẤP mô hình Động học Xe Ackermann (Ackermann Kinematics) & Quán tính phanh vật lý.

Chức năng:
  - Tích hợp Mô hình Động học Ackermann: Kết nối trực tiếp Góc lái delta với Tốc độ góc quay xe psi_dot = (v / L) * tan(delta).
  - Tích hợp Quãng đường Phanh Vật lý (Dynamic Braking Cushion): Tự động tính quãng đường trôi quán tính (v^2 / 2a_brake).
  - Tự động bẻ lái lượn vòng né tường (Steering Evasion) kết hợp phanh chủ động khi ở vận tốc cao.
  - Giải bài toán Quadratic Program (QP) thời gian thực bằng Scipy SLSQP / OSQP (< 1ms).
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
        fov_cutoff_deg: float = 75.0, # Góc quét phía trước xét vật cản (+/- độ)
        wheelbase: float = 0.39,      # Chiều dài cơ sở đo thực tế từ xe thật L (m)
        a_max_brake: float = 2.61,     # Gia tốc phanh tối đa thực tế đo từ xe thật (m/s^2)
        lat_accel_max: float = 4.5    # Gia tốc bám đường hướng tâm tối đa (m/s^2)
    ):
        self.d_min = d_min
        self.gamma = gamma
        self.v_max = v_max
        self.steer_max = steer_max
        self.slack_weight = slack_weight
        self.num_danger_rays = num_danger_rays
        self.fov_cutoff_rad = math.radians(fov_cutoff_deg)

        # --- Tham số Động học Xe Ackermann ---
        self.wheelbase = wheelbase
        self.a_max_brake = a_max_brake
        self.lat_accel_max = lat_accel_max

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

        # 1. Trích xuất các ràng buộc Ackermann Kinematic CBF từ LiDAR
        G_cbf, h_cbf = self._extract_cbf_constraints(v_nom, ranges, angles)

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

    def _extract_cbf_constraints(self, v_nom: float, ranges: np.ndarray, angles: np.ndarray):
        """
        Tạo ma trận G và vector h cho bất đẳng thức G * x <= h
        từ Động học Ackermann & Quán tính Phanh Vật lý.
        """
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

        v_curr = max(0.5, v_nom)

        for idx in danger_indices:
            r_i = float(valid_ranges[idx])
            phi_i = float(valid_angles[idx])

            # 1. Quãng đường trôi quán tính phanh vật lý: d_brake = (v * cos(phi_i))^2 / (2 * a_brake)
            d_brake = (v_curr * math.cos(phi_i))**2 / (2.0 * self.a_max_brake)

            # 2. Hàm Barrier Động Học: h_i = r_i - d_min - d_brake >= 0
            h_val = r_i - self.d_min - d_brake

            # 3. Đạo hàm theo mô hình Ackermann Kinematics:
            # - Tốc độ lao thẳng tiến: v * cos(phi_i)
            # - Tốc độ lượn vòng bẻ lái Ackermann: (v / L) * tan(delta) * (r_i * sin(phi_i))
            # Tuyến tính hóa theo [v, delta, slack]:
            g_v = math.cos(phi_i) * (1.0 + (v_curr * math.cos(phi_i)) / self.a_max_brake)
            g_steer = -(v_curr / self.wheelbase) * (r_i * math.sin(phi_i))

            G_list.append([g_v, g_steer, -1.0])
            h_list.append(self.gamma * max(0.01, h_val))

        return np.array(G_list), np.array(h_list)
