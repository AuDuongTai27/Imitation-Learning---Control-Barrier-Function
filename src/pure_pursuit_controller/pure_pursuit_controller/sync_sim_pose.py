#!/usr/bin/env python3
"""
sync_sim_pose.py
────────────────
Script tự động đọc vị trí xuất phát từ file waypoint.csv và cập nhật sx, sy, stheta trong sim.yaml (f1tenth_gym_ros).
Giúp đồng bộ 100% vị trí xe mô phỏng trùng khớp với waypoint được ghi từ Sim hoặc Xe thật.
"""

import os
import csv
import math
import sys
import yaml


def sync_pose(waypoint_path=None, sim_yaml_path=None):
    home = os.path.expanduser('~')

    # 1. Tìm file waypoint.csv
    if not waypoint_path:
        candidates = [
            "/sim_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv",
            os.path.join(home, "f1_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"),
            os.path.join(home, "f1_ws/src/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"),
            os.path.join(home, "danh_pp_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                waypoint_path = c
                break

    if not waypoint_path or not os.path.exists(waypoint_path):
        print(f"❌ Không tìm thấy file waypoint.csv!")
        return False

    # 2. Đọc dòng đầu tiên từ waypoint.csv
    pts = []
    with open(waypoint_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for r in reader:
            if len(r) >= 2:
                try:
                    pts.append([float(r[0]), float(r[1])])
                except ValueError:
                    pass

    if len(pts) < 2:
        print(f"❌ File waypoint {waypoint_path} không đủ dữ liệu (cần ít nhất 2 điểm)!")
        return False

    x0, y0 = pts[0][0], pts[0][1]
    x1, y1 = pts[1][0], pts[1][1]
    stheta = math.atan2(y1 - y0, x1 - x0)

    # 3. Tìm file sim.yaml
    if not sim_yaml_path:
        yaml_candidates = [
            "/sim_ws/src/f1tenth_gym_ros/config/sim.yaml",
            os.path.join(home, "f1_ws/src/f1tenth_gym_ros/config/sim.yaml"),
        ]
        for yc in yaml_candidates:
            if os.path.exists(yc):
                sim_yaml_path = yc
                break

    if not sim_yaml_path or not os.path.exists(sim_yaml_path):
        print(f"❌ Không tìm thấy file config/sim.yaml!")
        return False

    # 4. Đọc và Cập nhật sim.yaml
    try:
        with open(sim_yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        if 'bridge' in data and 'ros__parameters' in data['bridge']:
            params = data['bridge']['ros__parameters']
            params['sx'] = float(round(x0, 4))
            params['sy'] = float(round(y0, 4))
            params['stheta'] = float(round(stheta, 4))

            with open(sim_yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)

            print("=" * 60)
            print(" ✅ TỰ ĐỘNG ĐỒNG BỘ VỊ TRÍ XUẤT PHÁT CHO SIM THÀNH CÔNG")
            print(f" 📍 Waypoint CSV : {waypoint_path}")
            print(f" ⚙️ Config YAML  : {sim_yaml_path}")
            print(f" 🚀 Starting Pose: sx={x0:.4f}, sy={y0:.4f}, stheta={stheta:.4f} rad ({math.degrees(stheta):.1f}°)")
            print("=" * 60)
            return True
        else:
            print("❌ File sim.yaml không đúng định dạng F1TENTH bridge params!")
            return False

    except Exception as e:
        print(f"❌ Lỗi khi đọc/ghi sim.yaml: {e}")
        return False


if __name__ == '__main__':
    wp = sys.argv[1] if len(sys.argv) > 1 else None
    yaml_p = sys.argv[2] if len(sys.argv) > 2 else None
    sync_pose(wp, yaml_p)
