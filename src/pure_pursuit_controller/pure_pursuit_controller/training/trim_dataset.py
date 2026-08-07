#!/usr/bin/env python3
"""
trim_dataset.py
───────────────
Script dùng để cắt bớt / lọc thông minh dữ liệu file CSV Dataset (đặc biệt là file DAgger CSV).

Giải quyết vấn đề:
  - Dataset DAgger có quá nhiều mẫu bẻ lái gắt khẩn cấp (10.000 mẫu) khiến AI bị mất khả năng né từ xa mượt mà.
  - Script này hỗ trợ các chế độ cắt bớt:
    1. 'stride'   : Cắt lấy đều theo bước nhảy (Stride Subsampling - ví dụ cứ 3 dòng lấy 1 dòng).
    2. 'random'   : Lấy ngẫu nhiên N mẫu (hoặc theo tỷ lệ ratio % mong muốn).
    3. 'dedup'    : Loại bỏ các mẫu bẻ lái trùng lặp / giữ nguyên góc bẻ lái quá lâu (> K dòng).
    4. 'smart'    : Lọc thông minh giữ lại hỗn hợp mẫu né xa + né gần theo dải khoảng cách LiDAR min.

Cú pháp sử dụng:
  # Cắt lấy 3.000 mẫu ngẫu nhiên từ file dagger 10.000 mẫu:
  python3 trim_dataset.py --input dagger_dataset_real.csv --output dagger_trimmed.csv --mode random --count 3000

  # Cắt lấy 1/3 dữ liệu bằng bước nhảy (stride = 3):
  python3 trim_dataset.py --input dagger_dataset_real.csv --output dagger_trimmed.csv --mode stride --stride 3

  # Lọc bớt mẫu trùng lặp bẻ lái gắt giữ nguyên quá 5 dòng liên tiếp:
  python3 trim_dataset.py --input dagger_dataset_real.csv --output dagger_trimmed.csv --mode dedup --max_repeat 5

  # Lọc thông minh giữ 40% né xa (>2m), 40% né trung (1-2m), 20% cứu nguy (<1m):
  python3 trim_dataset.py --input dagger_dataset_real.csv --output dagger_trimmed.csv --mode smart --count 3500
"""

import os
import sys
import csv
import argparse
import numpy as np


def trim_stride(rows, header, stride=3):
    """Lấy đều theo bước nhảy stride (ví dụ stride=3 -> lấy dòng 0, 3, 6, 9...)"""
    trimmed = rows[::stride]
    print(f"✂️ [Stride={stride}] Cắt từ {len(rows)} mẫu xuống {len(trimmed)} mẫu.")
    return trimmed


def trim_random(rows, header, count=None, ratio=None, seed=42):
    """Lấy ngẫu nhiên N mẫu hoặc theo tỷ lệ ratio %"""
    np.random.seed(seed)
    total = len(rows)
    if count is not None:
        target_n = min(count, total)
    elif ratio is not None:
        target_n = int(total * ratio)
    else:
        target_n = total // 2

    indices = np.random.choice(total, size=target_n, replace=False)
    indices.sort()
    trimmed = [rows[i] for i in indices]
    print(f"🎲 [Random] Cắt ngẫu nhiên từ {total} mẫu xuống {len(trimmed)} mẫu (Tỷ lệ: {len(trimmed)/total*100:.1f}%).")
    return trimmed


def trim_dedup(rows, header, input_dim=60, max_repeat=5, steer_threshold=0.02):
    """Loại bỏ các mẫu bị trùng lặp góc bẻ lái quá K dòng liên tiếp"""
    steer_idx = input_dim + 1  # Cột steering_angle là cột cuối cùng
    trimmed = []
    repeat_count = 0
    last_steer = None

    for row in rows:
        try:
            steer = float(row[steer_idx])
        except (ValueError, IndexError):
            trimmed.append(row)
            continue

        if last_steer is not None and abs(steer - last_steer) < steer_threshold:
            repeat_count += 1
        else:
            repeat_count = 0
            last_steer = steer

        if repeat_count < max_repeat:
            trimmed.append(row)

    print(f"🧹 [De-duplicate] Loại bỏ các mẫu trùng lặp góc lái > {max_repeat} dòng. Cắt từ {len(rows)} xuống {len(trimmed)} mẫu.")
    return trimmed


def trim_smart(rows, header, input_dim=60, target_count=3500, seed=42):
    """
    Lọc thông minh: Phân bổ dữ liệu đều theo khoảng cách LiDAR nhỏ nhất (min_dist).
    Giúp mô hình vừa giữ được khả năng né xa (min_dist > 2.0m) vừa giữ được phản xạ cứu nguy (min_dist < 1.0m).
    """
    np.random.seed(seed)
    min_dists = []
    for row in rows:
        try:
            lidar_beams = [float(x) for x in row[:input_dim]]
            min_dists.append(min(lidar_beams))
        except ValueError:
            min_dists.append(5.0)

    min_dists = np.array(min_dists)

    # Chia dữ liệu làm 3 nhóm khoảng cách
    far_mask = min_dists >= 2.0         # Né từ xa (> 2m)
    mid_mask = (min_dists >= 1.0) & (min_dists < 2.0)  # Cua trung bình (1m - 2m)
    close_mask = min_dists < 1.0        # Cứu nguy gần (< 1m)

    far_indices = np.where(far_mask)[0]
    mid_indices = np.where(mid_mask)[0]
    close_indices = np.where(close_mask)[0]

    print(f"📊 Phân bố khoảng cách ban đầu:")
    print(f"   - Né xa (>= 2.0m) : {len(far_indices)} mẫu")
    print(f"   - Né trung (1-2m) : {len(mid_indices)} mẫu")
    print(f"   - Cứu nguy (< 1.0m): {len(close_indices)} mẫu")

    # Phân bổ ngân sách mẫu: 40% Né xa, 40% Né trung, 20% Cứu nguy
    n_far = min(len(far_indices), int(target_count * 0.40))
    n_mid = min(len(mid_indices), int(target_count * 0.40))
    n_close = min(len(close_indices), int(target_count * 0.20))

    selected_far = np.random.choice(far_indices, size=n_far, replace=False) if n_far > 0 else np.array([], dtype=int)
    selected_mid = np.random.choice(mid_indices, size=n_mid, replace=False) if n_mid > 0 else np.array([], dtype=int)
    selected_close = np.random.choice(close_indices, size=n_close, replace=False) if n_close > 0 else np.array([], dtype=int)

    all_selected = np.concatenate([selected_far, selected_mid, selected_close])
    all_selected.sort()

    trimmed = [rows[i] for i in all_selected]
    print(f"🧠 [Smart Trim] Đã cân bằng lại: Far={len(selected_far)}, Mid={len(selected_mid)}, Close={len(selected_close)}.")
    print(f"   Tổng cộng cắt từ {len(rows)} xuống {len(trimmed)} mẫu.")
    return trimmed


def main():
    parser = argparse.ArgumentParser(description="Script cắt bớt & lọc dữ liệu CSV Dataset cho DAgger")
    parser.add_argument('--input', type=str, required=True, help='Đường dẫn file CSV đầu vào')
    parser.add_argument('--output', type=str, required=True, help='Đường dẫn file CSV đầu ra sau khi cắt')
    parser.add_argument('--mode', type=str, choices=['stride', 'random', 'dedup', 'smart'], default='smart',
                        help='Chế độ cắt: stride (bước nhảy), random (ngẫu nhiên), dedup (lọc trùng), smart (lọc cân bằng khoảng cách)')
    parser.add_argument('--count', type=int, default=None, help='Số mẫu muốn lấy (dùng cho mode random / smart)')
    parser.add_argument('--ratio', type=float, default=None, help='Tỷ lệ mẫu muốn giữ lại (0.1 đến 0.9, dùng cho mode random)')
    parser.add_argument('--stride', type=int, default=3, help='Bước nhảy (dùng cho mode stride, mặc định 3)')
    parser.add_argument('--max_repeat', type=int, default=5, help='Số lần trùng bẻ lái tối đa được giữ lại (dùng cho mode dedup)')
    parser.add_argument('--input_dim', type=int, default=60, help='Số cột LiDAR (mặc định 60)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ File không tồn tại: {args.input}")
        sys.exit(1)

    with open(args.input, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    print(f"📂 Đã đọc file: {args.input} ({len(rows)} mẫu)")

    if args.mode == 'stride':
        trimmed_rows = trim_stride(rows, header, stride=args.stride)
    elif args.mode == 'random':
        trimmed_rows = trim_random(rows, header, count=args.count, ratio=args.ratio)
    elif args.mode == 'dedup':
        trimmed_rows = trim_dedup(rows, header, input_dim=args.input_dim, max_repeat=args.max_repeat)
    elif args.mode == 'smart':
        target_c = args.count if args.count is not None else 3500
        trimmed_rows = trim_smart(rows, header, input_dim=args.input_dim, target_count=target_c)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(trimmed_rows)

    print(f"✅ Đã lưu file mới thành công tại: {args.output}")


if __name__ == '__main__':
    main()
