#!/usr/bin/env python3
"""
check_csv.py (check_steering_distribution.py)
───────────────────────────────
Kiểm tra phân bố góc lái (angular_z) trong dataset DAgger để phát hiện mất cân bằng
(quá ít mẫu góc lái lớn) — nguyên nhân phổ biến khiến model MLP train bằng MSE
có xu hướng dự đoán gần 0 (đi thẳng) ở những khúc cua gắt hiếm gặp trong data.

Cách dùng:
    python check_csv.py --csv dagger_dataset_sim_90.csv
"""

import argparse
import csv
import os
import numpy as np


def main(csv_path, input_dim=None, n_bins=15):
    if not os.path.exists(csv_path):
        print(f"❌ File không tồn tại: {csv_path}")
        return

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if len(row) >= 2]

    if not rows:
        print(f"❌ File CSV rỗng hoặc không đúng định dạng: {csv_path}")
        return

    # Tự động phát hiện số chiều đặc trưng input nếu không truyền vào
    if input_dim is None or input_dim <= 0:
        input_dim = len(header) - 2

    # Luôn lấy 2 cột cuối cùng làm speed (linear_v) và steering_angle (angular_z)
    linear_v = np.array([float(row[-2]) for row in rows], dtype=np.float32)
    angular_z = np.array([float(row[-1]) for row in rows], dtype=np.float32)

    print(f"📄 Đang kiểm tra file: {csv_path}")
    print(f"📊 Tổng số mẫu: {len(angular_z)} | Số chiều input (Features): {input_dim}")
    print(f"angular_z (steering): min={angular_z.min():.4f}, max={angular_z.max():.4f}, "
          f"mean={angular_z.mean():.4f}, std={angular_z.std():.4f}")
    print(f"linear_v (speed):     min={linear_v.min():.4f}, max={linear_v.max():.4f}, "
          f"mean={linear_v.mean():.4f}, std={linear_v.std():.4f}\n")

    # Histogram phân bố góc lái
    counts, bin_edges = np.histogram(angular_z, bins=n_bins)
    max_count = counts.max()
    print("Phân bố góc lái (angular_z):")
    for i in range(n_bins):
        bar_len = int(50 * counts[i] / max_count) if max_count > 0 else 0
        pct = 100 * counts[i] / len(angular_z)
        print(f"  [{bin_edges[i]:+.3f}, {bin_edges[i+1]:+.3f}) : {counts[i]:6d} ({pct:5.1f}%) {'#' * bar_len}")

    # Cảnh báo nếu mất cân bằng nặng
    abs_steer = np.abs(angular_z)
    near_zero_pct = 100 * np.mean(abs_steer < 0.05)
    large_steer_pct = 100 * np.mean(abs_steer > 0.2)
    print(f"\n% mẫu gần như đi thẳng (|angular_z| < 0.05): {near_zero_pct:.1f}%")
    print(f"% mẫu cua gắt (|angular_z| > 0.2):             {large_steer_pct:.1f}%")

    if large_steer_pct < 5.0:
        print("\n⚠️  CẢNH BÁO: dưới 5% dữ liệu là cua gắt (|angular_z| > 0.2).")
        print("   Model MSE rất dễ bị 'kéo về 0' (đi thẳng) ở các khúc cua hiếm gặp này.")
        print("   -> Nên thu thêm dữ liệu DAgger tập trung vào đúng khúc cua gắt trên đường thật,")
        print("      hoặc dùng weighted loss / oversampling cho các mẫu góc lái lớn khi train lại.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='dagger_dataset_sim_5.csv')
    parser.add_argument('--input_dim', type=int, default=None, help='Số lượng cột input (mặc định tự động suy ra)')
    parser.add_argument('--bins', type=int, default=15)
    args = parser.parse_args()
    main(args.csv, args.input_dim, args.bins)