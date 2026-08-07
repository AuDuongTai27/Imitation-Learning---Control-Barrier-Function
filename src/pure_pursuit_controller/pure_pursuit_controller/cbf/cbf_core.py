#!/usr/bin/env python3
"""
cbf_core.py
───────────
Control Barrier Function (CBF-QP) safety filter implementation for F1TENTH.
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
        d_min: float = 0.35,          # Minimum safety distance (m)
        gamma: float = 2.0,           # CBF gain parameter
        v_max: float = 3.0,           # Velocity limit (m/s)
        steer_max: float = 0.41,      # Steering angle limit (rad)
        slack_weight: float = 1e4,    # Slack variable penalty weight
        num_danger_rays: int = 15,    # Number of nearest LiDAR beams for QP constraints
        fov_cutoff_deg: float = 10.0  # Frontal FOV cutoff (+/- deg)
    ):
        self.d_min = d_min
        self.gamma = gamma
        self.v_max = v_max
        self.steer_max = steer_max
        self.slack_weight = slack_weight
        self.num_danger_rays = num_danger_rays
        self.fov_cutoff_rad = math.radians(fov_cutoff_deg)

    def filter(self, u_nominal: np.ndarray, ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Filter nominal control command u_nominal = [v_nom, delta_nom] via CBF-QP"""
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
        """Fallback QP solver using SciPy SLSQP"""
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
        """Construct G matrix and h vector for CBF constraints G * x <= h"""
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
            # CBF Constraint: v * cos(phi_i) - slack <= gamma * (r_i - d_min)
            G_list.append([math.cos(phi_i), 0.0, -1.0])
            h_list.append(self.gamma * h_val)

        return np.array(G_list), np.array(h_list)
