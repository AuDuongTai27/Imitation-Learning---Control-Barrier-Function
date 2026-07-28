#!/usr/bin/env python3
"""
filter_dataset.py
─────────────────
Script cân bằng dữ liệu (Data Balancing) cho tập dữ liệu F1TENTH.
Lọc bớt các mẫu đi thẳng (góc lái gần bằng 0) để tránh làm loãng dữ liệu cua gắt,
giúp mô hình học nhạy bén hơn và không bị "lười" bẻ lái.
"""

import os
import csv
import random

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Danh sách các đường dẫn tìm kiếm của dataset (trong Docker và ngoài Host)
    potential_paths = [
        '/sim_ws/src/pure_pursuit_controller/pure_pursuit_controller/dagger_dataset_sim_4.csv',
        '/home/adt/f1_ws/src/pure_pursuit_controller/pure_pursuit_controller/dagger_dataset_sim_4.csv',
        os.path.join(current_dir, 'dagger_dataset_sim_4.csv')
    ]
    
    input_csv = None
    for path in potential_paths:
        if os.path.exists(path):
            input_csv = path
            break
            
    if input_csv is None:
        print("❌ Không tìm thấy file dữ liệu 'dagger_dataset_sim_4.csv' ở các đường dẫn:")
        for path in potential_paths:
            print(f"   - {path}")
        return
        
    # Lưu file đã lọc vào cùng thư mục với file gốc
    output_csv = input_csv.replace('dagger_dataset_sim_4.csv', 'dagger_dataset_sim_4_balanced.csv')
    
    print(f"📖 Đang đọc file dữ liệu gốc: {input_csv}...")
    
    header = None
    straight_samples = []
    corner_samples = []
    
    # Đọc dữ liệu
    with open(input_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Tìm cột góc lái (cột cuối cùng)
        steer_idx = len(header) - 1
        
        for row in reader:
            if not row:
                continue
            try:
                steer = float(row[steer_idx])
                # Ngưỡng góc lái nhỏ (khoảng dưới 5.7 độ / 0.1 rad)
                if abs(steer) < 0.1:
                    straight_samples.append(row)
                else:
                    corner_samples.append(row)
            except ValueError:
                continue
                
    total_original = len(straight_samples) + len(corner_samples)
    print(f"📊 Thống kê ban đầu:")
    print(f"   - Tổng số mẫu: {total_original}")
    print(f"   - Mẫu đi thẳng/cua nhẹ (|steer| < 0.1): {len(straight_samples)} ({len(straight_samples)/total_original*100:.1f}%)")
    print(f"   - Mẫu cua thực sự (|steer| >= 0.1): {len(corner_samples)} ({len(corner_samples)/total_original*100:.1f}%)")
    
    # Thực hiện lọc (Downsampling) dữ liệu đi thẳng
    # Giữ lại khoảng 15% lượng dữ liệu đi thẳng ngẫu nhiên
    keep_ratio = 0.15
    num_to_keep = int(len(straight_samples) * keep_ratio)
    
    random.seed(42) # Đảm bảo tính lặp lại
    kept_straight_samples = random.sample(straight_samples, num_to_keep)
    
    balanced_dataset = kept_straight_samples + corner_samples
    # Trộn đều lại dữ liệu
    random.shuffle(balanced_dataset)
    
    print(f"🔄 Đang ghi dữ liệu đã cân bằng ra file mới...")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(balanced_dataset)
        
    total_balanced = len(balanced_dataset)
    print(f"🎉 Đã tạo thành công file dữ liệu cân bằng tại:\n   👉 {output_csv}")
    print(f"📊 Thống kê sau cân bằng:")
    print(f"   - Tổng số mẫu mới: {total_balanced} (Giảm {100 - (total_balanced/total_original*100):.1f}% dung lượng dư thừa)")
    print(f"   - Mẫu đi thẳng giữ lại: {len(kept_straight_samples)} ({len(kept_straight_samples)/total_balanced*100:.1f}%)")
    print(f"   - Mẫu cua giữ lại: {len(corner_samples)} ({len(corner_samples)/total_balanced*100:.1f}%)")
    print(f"💡 Hướng dẫn: Bây giờ bạn hãy sửa đường dẫn tập train trong train.py thành file mới này và chạy train lại với khoảng 80-100 Epoch nhé!")

if __name__ == '__main__':
    main()
