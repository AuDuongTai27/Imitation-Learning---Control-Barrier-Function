#!/usr/bin/env python3
"""
combine_datasets.py
───────────────────
Script dùng để gộp 2 (hoặc nhiều) file CSV dataset (LiDAR + speed + steering_angle) lại với nhau thành 1 file CSV duy nhất.

Sử dụng:
  python3 combine_datasets.py --file1 dataset1.csv --file2 dataset2.csv --output combined_dataset.csv
  Hoặc chỉ cần chạy: python3 combine_datasets.py (sẽ hiển thị giao diện nhập file trực quan)
"""

import os
import sys
import csv
import argparse
import numpy as np


def resolve_path(path):
    if os.path.exists(path):
        return os.path.abspath(path)
    dirname, filename = os.path.split(path)
    # Thử tìm trong thư mục datasets/
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    alt1 = os.path.join(curr_dir, '..', 'datasets', filename)
    if os.path.exists(alt1):
        return os.path.abspath(alt1)
    alt2 = os.path.join(curr_dir, filename)
    if os.path.exists(alt2):
        return os.path.abspath(alt2)
    return path


def combine_csv_files(file1_path, file2_path, output_path, shuffle=False):
    f1 = resolve_path(file1_path)
    f2 = resolve_path(file2_path)

    if not os.path.exists(f1):
        print(f"❌ Error: File thứ nhất không tồn tại: {file1_path}")
        return False
    if not os.path.exists(f2):
        print(f"❌ Error: File thứ hai không tồn tại: {file2_path}")
        return False

    print("=" * 65)
    print(" 🔗 TIẾN HÀNH GỘP DỮ LIỆU DATASET CSV")
    print(f" 📂 File 1  : {f1}")
    print(f" 📂 File 2  : {f2}")
    print(f" 🎯 Output  : {output_path}")
    print("=" * 65)

    # Đọc file 1
    rows1 = []
    with open(f1, 'r', newline='') as f:
        reader = list(csv.reader(f))
        if len(reader) == 0:
            print(f"❌ Error: File {f1} rỗng!")
            return False
        header1 = reader[0]
        rows1 = reader[1:]

    # Đọc file 2
    rows2 = []
    with open(f2, 'r', newline='') as f:
        reader = list(csv.reader(f))
        if len(reader) == 0:
            print(f"❌ Error: File {f2} rỗng!")
            return False
        header2 = reader[0]
        rows2 = reader[1:]

    # Kiểm tra tính tương thích số cột giữa 2 file
    if len(header1) != len(header2):
        print(f"⚠️ CẢNH BÁO: Số lượng cột không khớp! (File 1 có {len(header1)} cột, File 2 có {len(header2)} cột)")
        confirm = input("Bạn có muốn tiếp tục gộp không? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Đã hủy quá trình gộp.")
            return False

    all_rows = rows1 + rows2
    print(f" 📊 Số mẫu File 1 : {len(rows1):,}")
    print(f" 📊 Số mẫu File 2 : {len(rows2):,}")
    print(f" 📈 Tổng cộng     : {len(all_rows):,} mẫu dữ liệu")

    # Xáo trộn dữ liệu ngẫu nhiên nếu được chọn
    if shuffle:
        np.random.shuffle(all_rows)
        print(" 🔀 Đã xáo trộn ngẫu nhiên (Shuffle) toàn bộ các dòng dữ liệu.")

    # Đảm bảo thư mục lưu file tồn tại
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Ghi ra file output
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header1)
        writer.writerows(all_rows)

    print("=" * 65)
    print(f" ✅ THÀNH CÔNG! Đã gộp và lưu {len(all_rows):,} mẫu vào file:")
    print(f" 💾 {os.path.abspath(output_path)}")
    print("=" * 65)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gộp 2 file CSV Dataset F1TENTH")
    parser.add_argument('--file1', type=str, default='', help='Đường dẫn tới file CSV thứ nhất')
    parser.add_argument('--file2', type=str, default='', help='Đường dẫn tới file CSV thứ hai')
    parser.add_argument('--output', type=str, default='', help='Đường dẫn file CSV kết quả sau khi gộp')
    parser.add_argument('--shuffle', action='store_true', help='Xáo trộn ngẫu nhiên dữ liệu sau khi gộp')

    args = parser.parse_args()

    f1 = args.file1
    f2 = args.file2
    out = args.output

    # Nếu không truyền qua CLI arguments, hiển thị giao diện nhập
    if not f1 or not f2 or not out:
        print("\n" + "═" * 65)
        print(" 🛠️  CÔNG CỤ GỘP 2 FILE DATASET CSV (MERGE CSV DATASETS)")
        print("═" * 65)
        try:
            if not f1:
                f1 = input("👉 Nhập đường dẫn File CSV 1 (vd: rrt_1.csv hoặc datasets/rrt_1.csv): ").strip()
            if not f2:
                f2 = input("👉 Nhập đường dẫn File CSV 2 (vd: rrt_2.csv hoặc datasets/rrt_2.csv): ").strip()
            if not out:
                out = input("👉 Nhập đường dẫn File CSV Output (vd: datasets/combined.csv): ").strip()
                if not out:
                    out = "datasets/combined_dataset.csv"
        except (KeyboardInterrupt, EOFError):
            print("\nĐã hủy.")
            sys.exit(0)

    combine_csv_files(f1, f2, out, shuffle=args.shuffle)
