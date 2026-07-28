#!/usr/bin/env python3
"""
train.py
────────
Script huấn luyện mô hình PyTorch Offline dùng cho cơ chế DAgger (Imitation Learning).
Tùy biến để chạy trực tiếp trên Google Colab với các file nằm cùng cấp thư mục.

Cải tiến so với bản gốc:
  1. Tự động phát hiện số tia LiDAR (input_dim) từ file CSV thay vì hardcode 60.
  2. Chia Train/Val theo BLOCK liên tục theo thời gian (giảm data leakage do LiDAR
     frame liền kề gần như giống hệt nhau) thay vì random theo từng dòng.
     Nếu CSV có cột 'episode', sẽ chia theo episode (chuẩn nhất).
  3. Chuẩn hóa target (linear_v, angular_z) để MSE không bị linear_v áp đảo.
  4. Log riêng MAE cho từng output (velocity vs steering) mỗi epoch.
  5. Early stopping dựa trên val loss, tránh train dư epoch.
  6. weight_decay + LR scheduler (ReduceLROnPlateau).
  7. Seed cố định để reproduce được kết quả.
  8. Đọc CSV bằng numpy thay vì vòng lặp Python thuần -> nhanh hơn nhiều.
"""

import os
import csv
import json
import numpy as np

# PyTorch Imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, Subset
except ImportError:
    raise ImportError("Thiếu thư viện PyTorch! Hãy cài đặt bằng lệnh: pip install torch torchvision")


SEED = 42


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- 1. Định nghĩa Mạng Neural (Đầu vào tự động tương thích với input_dim) ---
class DAggerMLP(nn.Module):
    def __init__(self, input_dim=90, output_dim=2, dropout=0.1):
        super(DAggerMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.network(x)


# --- 2. Dataset Custom cho DAgger ---
class DAggerDataset(Dataset):
    """
    Đọc CSV với input_dim cột LiDAR đầu + 2 cột cuối (linear_v, angular_z).
    Nếu CSV có cột tên 'episode' (hoặc 'run_id'/'run'), sẽ lưu lại để hỗ trợ
    chia train/val theo episode thay vì random theo dòng.

    Target được chuẩn hóa bằng (x - mean) / std, các giá trị mean/std được
    lưu lại trong self.target_mean / self.target_std để dùng lúc inference.
    """

    LIDAR_MAX_RANGE = 10.0  # m — chỉnh lại nếu range_max của sensor/sim khác

    def __init__(self, csv_path, input_dim=90):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Không tìm thấy file dataset tại: {csv_path}. Hãy tải file lên Colab!")

        self.input_dim = input_dim

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

        episode_col = None
        for cand in ('episode', 'run_id', 'run'):
            if cand in header:
                episode_col = header.index(cand)
                break

        raw = np.genfromtxt(csv_path, delimiter=',', skip_header=1, dtype=str)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        raw = raw[np.array([len(row) >= (input_dim + 2) for row in raw])]

        self.inputs = raw[:, :input_dim].astype(np.float32)
        self.targets = raw[:, input_dim:input_dim + 2].astype(np.float32)

        if episode_col is not None:
            self.episodes = raw[:, episode_col]
        else:
            self.episodes = None

        # Chuẩn hóa input LiDAR về [0, 1]
        self.inputs = self.inputs / self.LIDAR_MAX_RANGE

        # Chuẩn hóa target (z-score), lưu lại mean/std để inference dùng ngược lại
        self.target_mean = self.targets.mean(axis=0)
        self.target_std = self.targets.std(axis=0)
        self.target_std[self.target_std < 1e-6] = 1.0  # tránh chia 0
        self.targets_norm = (self.targets - self.target_mean) / self.target_std

        print(f"Loaded dataset from {csv_path}")
        print(f"Total samples: {len(self.inputs)}")
        print(f"Input shape (LiDAR beams): {self.inputs.shape} | Target shape: {self.targets.shape}")
        print(f"Target mean (v, w): {self.target_mean} | std: {self.target_std}")
        if episode_col is not None:
            print(f"Episode column detected: yes -> {header[episode_col]}")
        else:
            print("Episode column detected: no (dùng block-split theo thời gian)")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets_norm[idx]


def split_dataset(dataset, val_split=0.2, n_blocks=10, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(dataset)

    if dataset.episodes is not None:
        unique_eps = np.unique(dataset.episodes)
        rng.shuffle(unique_eps)
        n_val_eps = max(1, int(len(unique_eps) * val_split))
        val_eps = set(unique_eps[:n_val_eps])
        val_idx = np.where(np.isin(dataset.episodes, list(val_eps)))[0]
        train_idx = np.where(~np.isin(dataset.episodes, list(val_eps)))[0]
    else:
        block_edges = np.linspace(0, n, n_blocks + 1).astype(int)
        block_ids = np.arange(n_blocks)
        rng.shuffle(block_ids)
        n_val_blocks = max(1, int(n_blocks * val_split))
        val_blocks = set(block_ids[:n_val_blocks])

        val_mask = np.zeros(n, dtype=bool)
        for b in val_blocks:
            val_mask[block_edges[b]:block_edges[b + 1]] = True
        val_idx = np.where(val_mask)[0]
        train_idx = np.where(~val_mask)[0]

    return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist())


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_abs_err = np.zeros(2, dtype=np.float64)
    n = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_abs_err += torch.abs(outputs - targets).sum(dim=0).cpu().numpy()
            n += inputs.size(0)
    return total_loss / n, total_abs_err / n


# --- 3. Tiến trình Huấn luyện ---
def train_model(csv_path, save_path, epochs=100, batch_size=64, lr=0.001,
                 val_split=0.2, weight_decay=1e-5, patience=15, dropout=0.1):
    set_seed()

    # Tự động phát hiện số cột LiDAR (input_dim) từ header của file CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file dataset tại: {csv_path}. Hãy tải file lên trước!")

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
    
    # Đếm số lượng cột có tên bắt đầu bằng 'lidar_'
    lidar_cols = [h for h in header if h.startswith('lidar_')]
    input_dim = len(lidar_cols)
    if input_dim == 0:
        # Fallback nếu không có header dạng lidar_
        input_dim = len(header) - 2
        
    print(f"=== TỰ ĐỘNG PHÁT HIỆN HÌNH HỌC LIDAR ===")
    print(f"Số tia LiDAR (input_dim): {input_dim}")
    print(f"=========================================")

    dataset = DAggerDataset(csv_path, input_dim=input_dim)
    train_dataset, val_dataset = split_dataset(dataset, val_split=val_split)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DAggerMLP(input_dim=input_dim, output_dim=2, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"\nTraining on: {device}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr} | Weight decay: {weight_decay} | Patience: {patience}\n")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_dataset)

        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        val_mae_real = val_mae * dataset.target_std

        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} "
                  f"| MAE v: {val_mae_real[0]:.4f} m/s | MAE w: {val_mae_real[1]:.4f} rad/s | LR: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            dir_name = os.path.dirname(save_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            torch.save(model.state_dict(), save_path)

            # Lưu normalization stats cùng với model để inference dùng lại
            norm_path = os.path.splitext(save_path)[0] + '_norm.json'
            with open(norm_path, 'w') as f:
                json.dump({
                    'lidar_max_range': dataset.LIDAR_MAX_RANGE,
                    'target_mean': dataset.target_mean.tolist(),
                    'target_std': dataset.target_std.tolist(),
                }, f, indent=2)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping tại epoch {epoch+1} (val loss không cải thiện sau {patience} epoch liên tiếp).")
                break

    print(f"\nTraining Finished! Best Val Loss: {best_val_loss:.6f}")
    print(f"Best model weights saved to: {save_path}")
    print(f"Normalization stats saved to: {os.path.splitext(save_path)[0] + '_norm.json'}")
    print("LƯU Ý: khi inference thực tế, phải nhân output với target_std rồi cộng target_mean "
          "để quy đổi ngược về (linear_v, angular_z) thật.")


def resolve_dataset_path(path):
    if os.path.exists(path):
        return path
    dirname, filename = os.path.split(path)
    alt_path = os.path.join(dirname, 'datasets', filename)
    if os.path.exists(alt_path):
        return alt_path
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    alt_path2 = os.path.join(curr_dir, '..', 'datasets', filename)
    if os.path.exists(alt_path2):
        return os.path.abspath(alt_path2)
    return path

def resolve_model_path(path):
    dirname, filename = os.path.split(path)
    if not dirname:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(curr_dir, '..', 'models', filename)
    return path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train DAgger Model for F1TENTH")

    parser.add_argument('--csv', type=str,
                        default='rrt_2.csv',
                        help='Đường dẫn tới file CSV dataset')
    parser.add_argument('--model', type=str,
                        default='rrt_2.pth',
                        help='Đường dẫn lưu file trọng số mô hình (.pth)')
    parser.add_argument('--epochs', type=int, default=100, help='Số lượng epochs tối đa')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 regularization)')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience (số epoch)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    args, unknown = parser.parse_known_args()

    csv_resolved = resolve_dataset_path(args.csv)
    model_resolved = resolve_model_path(args.model)

    train_model(
        csv_path=csv_resolved,
        save_path=model_resolved,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=0.2,
        weight_decay=args.weight_decay,
        patience=args.patience,
        dropout=args.dropout,
    )
