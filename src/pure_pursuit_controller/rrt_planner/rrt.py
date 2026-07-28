#!/usr/bin/env python3
import numpy as np
import math
import random
import scipy.interpolate as interpolate
import csv  # <--- THÊM THƯ VIỆN CSV
import os   # <--- THÊM THƯ VIỆN OS

class treeNode():
    def __init__(self, x, y, is_root=False):
        self.x = x
        self.y = y
        self.is_root = is_root
        self.children = []
        self.parent = None
        self.cost = 0.0

class RRTStarAlgorithm():
    # --- CẬP NHẬT INIT ĐỂ NHẬN THÔNG SỐ MAP VÀ ĐƯỜNG DẪN FILE ---
    def __init__(self, start, goal, interations, collision_margin, steer_length, goal_tolerance, grid, 
                 resolution=0.05, origin=[-10.0, -10.0], output_file='rrt_path.csv'):
        self.start_node = treeNode(start[0], start[1], True)
        self.goal_node = treeNode(goal[0], goal[1])
        self.tree = [self.start_node]
        self.iterations = min(interations, 5000)
        self.grid = grid
        self.margin = collision_margin
        self.steer_length = steer_length
        self.goal_tolerance = goal_tolerance
        self.goalCosts = [10000]
        
        # Lưu các thông số để chuyển đổi tọa độ khi lưu CSV
        self.resolution = resolution
        self.origin = origin
        self.output_file = output_file
    # ------------------------------------------------------------

    def sample(self):
        free_x, free_y = np.where(self.grid == 0)
        if len(free_x) == 0: return None
        random_index = np.random.choice(len(free_x))
        return np.array([free_x[random_index], free_y[random_index]])

    def nearest(self, tree, sampled_point):
        nearest_node_idx = 0
        nearest_dist = 10000000
        for idx in range(len(tree)):
            dist = self.Eulidean_dist(tree[idx].x, tree[idx].y, sampled_point[0], sampled_point[1])
            if dist < nearest_dist:
                nearest_node_idx = idx
                nearest_dist = dist
        return nearest_node_idx

    def steer(self, nearest_node, sampled_point):
        new_node = treeNode(0, 0)
        dist = self.Eulidean_dist(nearest_node.x, nearest_node.y, sampled_point[0], sampled_point[1])
        if dist <= self.steer_length:
            new_node.x = sampled_point[0]
            new_node.y = sampled_point[1]
        else:
            new_node.x = int(nearest_node.x + (self.steer_length) / dist * (sampled_point[0] - nearest_node.x))
            new_node.y = int(nearest_node.y + (self.steer_length) / dist * (sampled_point[1] - nearest_node.y))
        return new_node

    def check_collision(self, nearest_node, new_node):
        x_end, y_end = int(new_node.x), int(new_node.y)
        x_start, y_start = int(nearest_node.x), int(nearest_node.y)

        dy = abs(y_end - y_start); dx = abs(x_end - x_start)
        sx = 1 if x_start < x_end else -1
        sy = 1 if y_start < y_end else -1
        error = dx - dy

        def is_within_bounds(x, y):
            return 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]

        def is_in_collision(x, y):
            for i in range(-self.margin, self.margin + 1):
                for j in range(-self.margin, self.margin + 1):
                    xi, yj = x + i, y + j
                    if (is_within_bounds(xi, yj) and self.grid[xi, yj] == 100) or (is_within_bounds(xi, yj) and self.grid[xi, yj] == -1):
                        return True
            return False

        while (x_start != x_end) or (y_start != y_end):
            if not is_within_bounds(x_start, y_start) or self.grid[x_start, y_start] == 100 or is_in_collision(x_start, y_start):
                return True
            e2 = 2 * error
            if e2 > -dy: error -= dy; x_start += sx
            if e2 < dx: error += dx; y_start += sy

        return is_in_collision(x_end, y_end)

    def is_goal(self, latest_added_node, goal_x, goal_y):
        return self.Eulidean_dist(latest_added_node.x, latest_added_node.y, goal_x, goal_y) <= self.goal_tolerance

    def find_path_2(self, latest_added_node):
        path = []
        current_node = latest_added_node
        while current_node is not None:
            if current_node == current_node.parent: break
            path.append(current_node)
            if current_node.is_root: break
            current_node = current_node.parent
        path.reverse()
        return path

    def cost(self, node):
        return 0.0 if node.is_root else node.cost

    def line_cost(self, n1, n2):
        return math.sqrt(pow((n1.x - n2.x), 2) + pow((n1.y - n2.y), 2))

    def near(self, tree, node):
        neighborhood = []
        for node_ in tree:
            if node_ != node:
                dist = self.Eulidean_dist(node_.x, node_.y, node.x, node.y)
                if dist <= self.steer_length * 2.0:
                    neighborhood.append(node_)
        return neighborhood

    def Eulidean_dist(self, x1, y1, x2, y2):
        return math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))

    # ==========================================
    # HÀM LÀM ĐẸP ĐƯỜNG (POST PROCESSING) & LƯU CSV
    # ==========================================
    def post_processing(self, path_nodes):
        if not path_nodes or len(path_nodes) < 3: return path_nodes

        # 1. Pruning
        pruned_path = self.pruning_path(path_nodes)

        # 2. Resampling
        resampled_path = self.resample_path(pruned_path, step=0.3)

        # 3. Smoothing
        smooth_coords = self.smoothing_path(resampled_path)

        # 4. Safety Check
        final_coords = []
        if self.is_path_safe(smooth_coords):
            final_coords = smooth_coords
        else:
            print("Smooth path hit obstacle! Fallback to pruned path.")
            final_coords = [(n.x, n.y) for n in pruned_path]

        # Chuyển đổi thành list treeNode
        final_nodes = [treeNode(x, y) for x, y in final_coords]

        # --- GỌI HÀM LƯU CSV TẠI ĐÂY ---
        # Sau khi đã có đường đi đẹp nhất, ta tiến hành lưu file
        self.save_path_to_csv(final_nodes)
        # -------------------------------

        return final_nodes

    def save_path_to_csv(self, path_nodes):
        """ Hàm chuyển đổi Grid -> World và lưu file CSV """
        try:
            # Mở rộng đường dẫn home (~) nếu có
            full_path = os.path.expanduser(self.output_file)
            print(f"[RRT] Saving path to {full_path}...")
            
            with open(full_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Duyệt qua các node để tính toán và lưu
                for i, node in enumerate(path_nodes):
                    # 1. Chuyển đổi Grid (ô) -> World (mét)
                    # Công thức: world = grid * resolution + origin
                    wx = node.x * self.resolution + self.origin[0]
                    wy = node.y * self.resolution + self.origin[1]
                    
                    # 2. Tính Yaw (Góc hướng)
                    yaw = 0.0
                    if i < len(path_nodes) - 1:
                        next_node = path_nodes[i+1]
                        next_wx = next_node.x * self.resolution + self.origin[0]
                        next_wy = next_node.y * self.resolution + self.origin[1]
                        yaw = math.atan2(next_wy - wy, next_wx - wx)
                    
                    # 3. Vận tốc (Giả định để Pure Pursuit chạy được)
                    # Chạy thẳng: 2.0 m/s, Cua: 1.0 m/s
                    velocity = 1.5 
                    
                    # Ghi: x, y, velocity, yaw
                    writer.writerow([f"{wx:.4f}", f"{wy:.4f}", f"{velocity:.2f}", f"{yaw:.4f}"])
            
            print("[RRT] CSV Saved Successfully!")
            
        except Exception as e:
            print(f"[RRT] Error saving CSV: {e}")

    def pruning_path(self, path):
        if len(path) < 3: return path
        pruned_path = [path[0]]
        current_idx = 0
        while current_idx < len(path) - 1:
            for lookahead_idx in range(len(path) - 1, current_idx, -1):
                if not self.check_collision(path[current_idx], path[lookahead_idx]):
                    pruned_path.append(path[lookahead_idx])
                    current_idx = lookahead_idx
                    break
                if lookahead_idx == current_idx + 1:
                    pruned_path.append(path[current_idx + 1])
                    current_idx += 1
        return pruned_path

    def resample_path(self, path_nodes, step=0.3):
        if len(path_nodes) < 2: return path_nodes
        new_path = []
        for i in range(len(path_nodes) - 1):
            n1 = path_nodes[i]
            n2 = path_nodes[i + 1]
            dist = math.hypot(n2.x - n1.x, n2.y - n1.y)
            new_path.append(n1)
            if dist > step:
                num_segments = int(dist / step)
                for j in range(1, num_segments):
                    alpha = j / num_segments
                    new_x = n1.x + (n2.x - n1.x) * alpha
                    new_y = n1.y + (n2.y - n1.y) * alpha
                    new_path.append(treeNode(new_x, new_y))
        new_path.append(path_nodes[-1])
        return new_path

    def smoothing_path(self, path_nodes):
        x = [node.x for node in path_nodes]
        y = [node.y for node in path_nodes]
        x_u, y_u = [x[0]], [y[0]]
        for i in range(1, len(x)):
            if abs(x[i] - x[i - 1]) > 0.01 or abs(y[i] - y[i - 1]) > 0.01:
                x_u.append(x[i])
                y_u.append(y[i])
        if len(x_u) < 3: return list(zip(x_u, y_u))
        try:
            tck, u = interpolate.splprep([x_u, y_u], k=3, s=0.2)
            u_fine = np.linspace(0, 1, num=len(x_u) * 3)
            x_fine, y_fine = interpolate.splev(u_fine, tck)
            return list(zip(x_fine, y_fine))
        except:
            return list(zip(x_u, y_u))

    def is_path_safe(self, path_coords):
        for i in range(len(path_coords) - 1):
            p1 = treeNode(path_coords[i][0], path_coords[i][1])
            p2 = treeNode(path_coords[i + 1][0], path_coords[i + 1][1])
            if self.check_collision(p1, p2):
                return False
        return True