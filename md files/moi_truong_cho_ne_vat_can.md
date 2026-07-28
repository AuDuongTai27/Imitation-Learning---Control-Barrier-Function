# Kế hoạch: DAgger với RRT Expert + Obstacle Spawning thông minh

## Bối cảnh

- **Hiện tại:** `real.onnx` đã học từ Pure Pursuit expert → lái được track trơn, nhưng không biết né vật cản.
- **Mục tiêu:** Dùng DAgger với RRT làm expert để model học né vật cản.
- **Vấn đề cốt lõi:** Làm sao spawn obstacle hợp lý để xe thực sự học được cách né, không phải chỉ đứng yên hoặc bị block?

---

## Phân tích vấn đề Obstacle Spawning

### Tại sao spawn ngẫu nhiên không hiệu quả?

```
[Track]
  ────────────────────────────────────
  Obstacle quá xa  → xe chưa thấy, RRT không kích hoạt
  Obstacle giữa đường → RRT block hoàn toàn, không có path
  Obstacle ngoài track → vô nghĩa, AI không học gì
  Obstacle quá gần → xe phản ứng không kịp, crash
  ────────────────────────────────────
```

### Điều kiện để obstacle "có ích" cho training:

1. **Nằm trên hoặc gần track** → AI phải đối mặt với nó
2. **Không chặn hoàn toàn** → RRT vẫn tìm được đường vòng
3. **Đủ xa phía trước** → xe có đủ thời gian/không gian để né (tối thiểu 1.5–2m)
4. **Vị trí có tính ngẫu nhiên có kiểm soát** → model học được tổng quát, không học thuộc lòng vị trí cố định

---

## Đề xuất Chiến lược Spawn Obstacle

### Chiến lược: Track-Guided Perpendicular Spawning

**Ý tưởng:** Thay vì spawn hoàn toàn ngẫu nhiên, lấy điểm dọc theo waypoint CSV và offset vuông góc với hướng track.

```
             ↑ hướng track
  ──[W_i-1]──[W_i]──[W_i+1]──
                │
                │ offset ±d vuông góc
                ▼
           [Obstacle]  ← nằm cạnh track, không chặn hoàn toàn
```

**Tham số điều chỉnh:**
| Tham số | Gợi ý | Ý nghĩa |
|---------|-------|---------|
| `spawn_distance_ahead` | 3–5m | Cách xa xe bao nhiêu mét thì spawn |
| `perpendicular_offset` | 0.0–0.3m | Lệch trái/phải so với tim đường (0 = giữa đường, 0.3 = bên lề) |
| `obstacle_radius` | 0.25–0.35m | Bán kính — đủ to để RRT phải né, nhưng đủ hẹp để có đường vòng |
| `respawn_interval` | 5–10s | Sau bao lâu thì dời vị trí obstacle |

**Tại sao offset nhỏ (0.0–0.3m)?**
- Track F1TENTH thường rộng ~1m
- Obstacle bán kính 0.3m + offset 0.3m = vẫn còn ~0.4m để xe đi qua
- RRT có thể tìm đường, AI học được cách lệch sang bên

---

## Kiến trúc DAgger Loop đề xuất

```
Vòng lặp DAgger (mỗi iteration ~10–20 phút sim):

┌─────────────────────────────────────────────────────────┐
│ 1. SETUP                                                │
│    - Load model hiện tại (lần đầu: real.onnx)          │
│    - Spawn obstacle theo chiến lược track-guided        │
│    - Bắt đầu thu thập data                              │
├─────────────────────────────────────────────────────────┤
│ 2. CHẠY SIM                                             │
│    AI lái xe (dùng model hiện tại)                      │
│         │                                               │
│         ├─ RRT luôn tính song song trong nền            │
│         │                                               │
│         ├─ Khi AI gặp obstacle (LiDAR < threshold):     │
│         │       → RRT expert tiếp quản                  │
│         │       → GHI DATA: [scan → expert_cmd]         │
│         │                                               │
│         └─ Khi đường thông thoáng:                      │
│               → AI lái, không ghi data                  │
├─────────────────────────────────────────────────────────┤
│ 3. RETRAIN                                              │
│    - Gộp dataset cũ (Pure Pursuit) + dataset mới (RRT) │
│    - Train lại model với train.py                       │
│    - Export .onnx mới                                   │
├─────────────────────────────────────────────────────────┤
│ 4. ĐÁNH GIÁ & LẶP LẠI                                  │
│    - Chạy model mới, so sánh với trước                  │
│    - Nếu tốt hơn → lấy làm base cho iteration tiếp     │
│    - Nếu tệ hơn → giữ model cũ, điều chỉnh tham số     │
└─────────────────────────────────────────────────────────┘
```

---

## Điều kiện kích hoạt RRT Expert (thay vì dùng góc lái)

Khác với bản DAgger cũ (so sánh góc lái AI vs Expert), đề xuất kích hoạt expert **dựa trên nguy hiểm LiDAR**:

```
min(lidar_front_beams) < danger_threshold (ví dụ 0.8m)
→ RRT expert tiếp quản
→ Ghi data
```

**Tại sao tốt hơn?**
- Không cần chờ AI lệch xa mới can thiệp
- Expert can thiệp đúng lúc — khi xe THỰC SỰ gần obstacle
- Data ghi được chính xác là các quyết định né tránh của RRT

---

## Các file cần tạo/sửa

### [NEW] `auto_obstacle_spawner.py`
Node tự động spawn và dịch chuyển obstacle theo track-guided strategy.
- Đọc waypoint CSV
- Tính vị trí spawn = waypoint phía trước xe + offset vuông góc
- Spawn và tự dịch chuyển sau `respawn_interval` giây

### [NEW] `dagger_rrt_collector.py`
Node thu thập data DAgger với trigger LiDAR-based.
- Subscribe `/scan` + `/drive` (lệnh expert từ RRT)
- Ghi data khi `min(front_scan) < danger_threshold`
- Output CSV giống format `dagger_dataset_sim.csv`

### [MODIFY] `obstacle_spawner.py` (tùy chọn)
Hoặc mở rộng node hiện tại để hỗ trợ thêm chế độ auto-spawn bên cạnh manual click.

---

## Curriculum học dần (nên áp dụng)

> [!TIP]
> Không nên cho model học tất cả độ khó ngay từ đầu. Học theo curriculum giúp model hội tụ nhanh và ổn định hơn.

| Giai đoạn | Số obstacle | Offset | `respawn_interval` | Mục tiêu |
|-----------|-------------|--------|--------------------|---------|
| 1 | 1 | 0.2m (bên lề) | 15s | Model nhận ra và né 1 obstacle đơn giản |
| 2 | 1 | 0.0m (giữa) | 10s | Né obstacle giữa đường |
| 3 | 2 | ngẫu nhiên | 8s | Né nhiều obstacle |
| 4 | 3+ | ngẫu nhiên | 5s | Tổng quát hóa |

---

## Open Questions — Cần xác nhận trước khi code

> [!IMPORTANT]
> **Q1:** RRT expert trong sim (`hybrid_planner_sim.py`) publish lệnh lái qua topic nào? `/drive` hay topic khác?
> Cần biết để DAgger collector đọc đúng nguồn lệnh chuyên gia.

> [!IMPORTANT]
> **Q2:** Khi chạy DAgger, AI và RRT có cùng publish `/drive` không?
> Cần xác định cơ chế chuyển giao — ai publish `/drive` tại bất kỳ thời điểm nào.

> [!IMPORTANT]
> **Q3:** `danger_threshold` bao nhiêu mét là hợp lý?
> Với tốc độ sim hiện tại (ví dụ 2–3 m/s) và brake distance, phải đủ xa để RRT kịp tính path.

> [!NOTE]
> **Q4:** Dataset hiện tại (`real.onnx` được train từ) có bao nhiêu samples?
> Để ước lượng cần thêm bao nhiêu samples RRT để không bị catastrophic forgetting.
