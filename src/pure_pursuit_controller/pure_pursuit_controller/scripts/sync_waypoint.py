#!/usr/bin/env python3
"""
sync_waypoint.py
────────────────
Script tự động copy/đồng bộ file f1tenth_waypoint.csv từ Máy Host trực tiếp vào DOCKER CONTAINER
và các vị trí trong Workspace, đồng thời tự động cập nhật vị trí xuất phát trong sim.yaml.

Cú pháp:
  python3 sync_waypoint.py [đường_dẫn_file_csv_nguồn]

Ví dụ:
  python3 sync_waypoint.py
  python3 sync_waypoint.py ~/Desktop/f1tenth_waypoint.csv
"""

import os
import sys
import shutil
import subprocess
from sync_sim_pose import sync_pose


def get_running_docker_containers():
    """Tìm các docker container đang chạy liên quan tới sim / f1tenth"""
    try:
        res = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True
        )
        containers = [c.strip() for c in res.stdout.splitlines() if c.strip()]
        # Ưu tiên các container chứa f1tenth hoặc sim
        sim_containers = [c for c in containers if 'f1tenth' in c.lower() or 'sim' in c.lower()]
        return sim_containers if sim_containers else containers
    except Exception:
        return []


def sync_waypoint_file(src_path=None):
    home = os.path.expanduser('~')

    # Các vị trí nguồn ưu tiên nếu không truyền src_path
    if not src_path:
        possible_sources = [
            os.path.join(home, "Desktop/f1tenth_waypoint.csv"),
            os.path.join(home, "f1tenth_waypoint.csv"),
            os.path.join(home, "f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/scripts/f1tenth_waypoint.csv"),
            os.path.join(home, "f1_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"),
            os.path.join(home, "f1_ws/src/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"),
        ]
        for s in possible_sources:
            if os.path.exists(s):
                src_path = s
                break

    if not src_path or not os.path.exists(src_path):
        print(f"❌ Không tìm thấy file CSV nguồn! Vui lòng chỉ định đường dẫn file CSV.")
        return False

    print("=" * 65)
    print(f" 📂 NGUỒN CSV HOST: {src_path}")

    # 1. Copy tới các vị trí trên máy Host
    target_destinations_host = [
        os.path.join(home, "f1_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"),
        os.path.join(home, "f1_ws/src/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"),
        os.path.join(home, "f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/scripts/f1tenth_waypoint.csv"),
    ]

    copied_host_count = 0
    print(" 🔄 1. Copy file waypoint tới Workspace Host...")
    for dst in target_destinations_host:
        try:
            if os.path.exists(dst) and os.path.samefile(src_path, dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src_path, dst)
            print(f"   -> [Host] Copy thành công sang: {dst}")
            copied_host_count += 1
        except Exception as e:
            pass

    # 2. Copy trực tiếp vào DOCKER CONTAINER bằng `docker cp`
    containers = get_running_docker_containers()
    docker_copied_count = 0
    if containers:
        print(" 🐳 2. Đồng bộ trực tiếp vào DOCKER CONTAINER...")
        docker_target_paths = [
            "/sim_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv",
            "/sim_ws/src/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv",
            "/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/scripts/f1tenth_waypoint.csv",
        ]
        for container in containers:
            for d_path in docker_target_paths:
                try:
                    # Đảm bảo thư mục cha tồn tại trong docker
                    parent_dir = os.path.dirname(d_path)
                    subprocess.run(["docker", "exec", container, "mkdir", "-p", parent_dir], capture_output=True)
                    
                    # thực hiện docker cp
                    cp_res = subprocess.run(["docker", "cp", src_path, f"{container}:{d_path}"], capture_output=True, text=True)
                    if cp_res.returncode == 0:
                        print(f"   -> [Docker '{container}'] Copy thành công sang: {d_path}")
                        docker_copied_count += 1
                except Exception as e:
                    pass
    else:
        print(" ⚠️  Không tìm thấy Docker container nào đang chạy (sẽ bỏ qua docker cp).")

    print(f" ✅ Đã copy xong ({copied_host_count} vị trí Host, {docker_copied_count} vị trí Docker).")
    print("=" * 65)

    # 3. Đồng bộ vị trí xuất phát sim.yaml trên Host
    sync_pose(waypoint_path=src_path)

    # 4. Nếu có Docker container đang chạy, chạy sync_sim_pose bên trong Docker luôn
    if containers:
        for container in containers:
            try:
                print(f" ⚙️ Đang đồng bộ sim.yaml bên trong Docker container '{container}'...")
                subprocess.run(
                    ["docker", "exec", container, "python3", "/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/scripts/sync_sim_pose.py"],
                    capture_output=True, text=True
                )
            except Exception:
                pass

    return True


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else None
    sync_waypoint_file(src)

