#!/usr/bin/env python3
import os


def resolve_waypoint_path(given_path):
    if given_path and os.path.exists(given_path):
        return given_path
    
    home = os.path.expanduser('~')
    candidates = [
        # Đường dẫn bên trong Docker Sim Container
        "/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/f1tenth_waypoint.csv",
        "/sim_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv",
        "/sim_ws/src/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv",
        # Đường dẫn ngoài máy thật Host
        os.path.join(home, "f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/f1tenth_waypoint.csv"),
        os.path.join(home, "f1_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"),
        os.path.join(home, "f1_ws/src/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"),
        os.path.join(home, "Desktop/f1tenth_waypoint.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return given_path

