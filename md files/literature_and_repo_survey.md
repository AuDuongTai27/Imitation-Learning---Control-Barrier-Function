# TỔNG HỢP CÁC BÀI BÁO RESEARCH & GITHUB REPO LIÊN QUAN (CẬP NHẬT LẦN 2)

**Đề tài:** *Ứng dụng Imitation Learning (MPD, DAgger) + Control Barrier Functions (CBF) Sim-to-Real trên hệ thống xe F1TENTH ROS 2.*

---

## 1. Các Bài báo Nghiên cứu Khoa học (Scientific Research Papers)

### 1.1. Nhóm bài báo về Imitation Learning (DAgger/MPD) & F1TENTH
1. **"A Benchmark Comparison of Imitation Learning-based Control Policies for Autonomous Racing"** (Sun et al., 2023 - UPenn mLAB)
   * **Nội dung:** Đánh giá benchmark các phương pháp Imitation Learning (Behavioral Cloning, Human-Gated DAgger - HG-DAgger) và Reinforcement Learning trên F1TENTH.
   * **Điểm tương đồng:** Sử dụng DAgger để khắc phục hiện tượng trôi phân phối dữ liệu (compounding error) và dùng bộ điều khiển chuyên gia để sinh dữ liệu mẫu.
   * **Điểm cải tiến của bạn:** Bài của UPenn chỉ dùng IL thuần túy (Black-box), **không có bộ lọc an toàn CBF**. Bài của bạn bổ sung **bộ lọc CBF-QP** đảm bảo 0% va chạm trên xe thật và áp dụng **Co-training Mixed Sim-to-Real Dataset (20k Sim + 6k Real)**.

2. **"MEGA-DAgger: Imitation Learning from Multiple Imperfect Experts"** (Hanbaliq et al.)
   * **Nội dung:** Mở rộng DAgger cho phép học từ nhiều nguồn expert khác nhau trên F1TENTH (bao gồm cả các nguồn expert có nhiễu hoặc không hoàn hảo).
   * **Điểm tương đồng:** Cách tiếp cận xử lý dữ liệu đa nguồn tương đồng với việc bạn kết hợp dữ liệu mô phỏng và thực tế.

3. **"End2Race: Efficient End-to-End Imitation Learning for Real-Time F1Tenth Racing"** (2025)
   * **Nội dung:** Triển khai IL thời gian thực trên F1TENTH với kiến trúc mạng nhẹ, tối ưu hóa tốc độ tính toán suy luận (inference speed) trên máy tính nhúng.

---

### 1.2. Nhóm bài báo về Control Barrier Functions (CBF) & Safety Shields (2023 - 2026)
1. **"End-to-End Imitation Learning with Safety Guarantees using Control Barrier Functions"** (Ryan K. Cosner, Yisong Yue, Aaron D. Ames - Caltech)
   * **Nội dung:** Tích hợp trực tiếp chính sách nơ-ron Imitation Learning với Control Barrier Function (CBF) để đưa ra cam kết an toàn toán học (Input-to-State Safety - ISSf) cho xe tự hành.
   * **So sánh:** Caltech chứng minh lý thuyết toán cho Vision End-to-End. Bài của bạn mang tính **thực nghiệm ứng dụng trên F1TENTH ROS 2 thực tế**, dùng dữ liệu LiDAR + Odometry và Co-training Sim-to-Real.

2. **"ConBaT: Control Barrier Transformer for Safety-Critical Policy Learning"** (2024-2025)
   * **Nội dung:** Ứng dụng mô hình Transformer kết hợp CBF để chọn hành động an toàn trong điều khiển robot và F1TENTH.
   * **So sánh:** ConBaT dùng kiến trúc Transformer phức tạp. Bài của bạn dùng **MLP nhẹ + CBF-QP Solver (<3ms latency)** tối ưu hóa cho phần cứng nhúng Jetson/NUC.

3. **"A Predictive Safety Filter for Learning-Based Racing Control"** (Shengfan Cao, Francesco Borrelli - UC Berkeley)
   * **Nội dung:** Xây dựng bộ lọc an toàn (Predictive Safety Filter via MPC) bọc ngoài chính sách học máy trong đua xe tự hành.
   * **So sánh:** Bài của Berkeley dùng MPC (tốn chi phí tính toán). Bài của bạn dùng **CBF-QP cơ bản**, tính toán cực nhanh thời gian thực.

4. **"Collision Cone Control Barrier Functions (C3BF) for Dynamic Obstacle Avoidance"** (2024-2026)
   * **Nội dung:** Phát triển CBF dạng nón va chạm để xử lý vật cản động trên các robot di động dạng Ackermann (tương tự F1TENTH).

---

## 2. Các Repository GitHub Mã Nguồn Mở (Relevant GitHub Repositories)

| Repository Name | GitHub Link | Main Framework / Tech | How it relates to your project |
| :--- | :--- | :--- | :--- |
| **f1tenth_il** (UPenn) | [mlab-upenn/f1tenth_il](https://github.com/mlab-upenn/f1tenth_il) | PyTorch, DAgger, HG-DAgger | Repo chuẩn về DAgger & IL trên F1TENTH của ĐH Pennsylvania. Có sẵn pipeline huấn luyện và đánh giá. |
| **f1tenth-MEGA-DAgger** | [derekhanbaliq/f1tenth-MEGA-DAgger](https://github.com/derekhanbaliq/f1tenth-MEGA-DAgger) | ROS 2 Foxy/Humble, DAgger | Bộ code ROS 2 hoàn chỉnh triển khai DAgger cho F1TENTH, tương thích với kiến trúc ROS 2 của bạn. |
| **CBFKit** | [bardhh/cbfkit](https://github.com/bardhh/cbfkit) | Python, JAX, ROS 2, CBF-QP | Thư viện CBF mã nguồn mở hỗ trợ Kinematic Bicycle Model và QP Solver cho ROS 2. |
| **safe_control** | [tkkim-robot/safe_control](https://github.com/tkkim-robot/safe_control) | ROS 2, CBF-QP, Bicycle Model | Code mẫu triển khai bộ lọc an toàn CBF-QP cho robot di động và xe Ackermann. |
| **CBF_QP_safety_filter** | [shaoanlu/CBF_QP_safety_filter](https://github.com/shaoanlu/CBF_QP_safety_filter) | Python, CVXOPT, LiDAR processing | Lớp lọc CBF đơn giản dùng dữ liệu LiDAR để giới hạn tốc độ và tránh va chạm. |
| **f1tenth_system** | [f1tenth/f1tenth_system](https://github.com/f1tenth/f1tenth_system) | ROS 2, Pure Pursuit, RRT* | Repo chính thức của F1TENTH chứa các gói ROS 2 cho Pure Pursuit và RRT planner. |

---

## 3. Bảng So Sánh Chi Tiết: Bài Báo Của Bạn vs. Nghiên Cứu Hiện Có

| Tiêu chí | Caltech (Cosner 2022) | UPenn (Sun 2023) | UC Berkeley (Cao 2023) | **Bài báo của BẠN (Proposed Work)** |
| :--- | :--- | :--- | :--- | :--- |
| **Định hướng chính** | Chứng minh Lý thuyết toán | Benchmark DAgger | MPC Safety Filter | **Ứng dụng Thực nghiệm F1TENTH ROS 2** |
| **Cơ chế an toàn (Safety)** | CBF Lý thuyết | ❌ Không có (Dễ đâm đụng) | MPC (Nặng tính toán) | **CBF-QP Thời gian thực (Nhẹ, <3ms)** |
| **Chuyên gia (Expert Generator)** | Thuật toán giả định | Pure Pursuit | MPC | **Pure Pursuit + RRT** (Bám làn & né vật cản) |
| **Giải quyết Sim-to-Real** | Mô phỏng | Thuần Sim / Real riêng | Mô phỏng | **Co-training Mixed Dataset (20k Sim + 6k Real)** |
| **Nền tảng thực nghiệm** | Xe mô hình nhỏ | F1TENTH Sim/Real | Simulator | **Xe F1TENTH phần cứng thật + ROS 2** |

---

## 4. Tóm tắt Giá trị Thêm (Value Proposition) cho Bài Báo của Bạn

Khi viết bài báo cáo, bạn nên nhấn mạnh 3 luận điểm vàng (Golden Value Propositions):
1. **Tiết kiệm 70% chi phí thu thập dữ liệu real:** Chỉ cần **6,000 mẫu real** kết hợp với **20,000 mẫu sim** trong quy trình Co-training giúp xe thích nghi với nhiễu thực tế mà không cần chạy thu thập hàng vạn mẫu trên xe thật.
2. **Lưới an toàn kép (Dual Safety Net):**
   * *Mức Planner (Off-line):* RRT Expert đảm bảo bộ dữ liệu huấn luyện không bị ô nhiễm bởi các hành vi đâm đụng.
   * *Mức Execution (On-line):* CBF-QP Solver làm màng lọc an toàn cứng, can thiệp điều chỉnh tốc độ/góc lái khi IL đưa ra lệnh lỗi.
3. **Triển khai ROS 2 thời gian thực tính bằng miligiây:** Mạng MLP kết hợp CBF-QP đạt độ trễ tổng cộng dưới 5ms, đảm bảo chạy tốt trên máy tính nhúng Jetson Orin / NUC ở tốc độ xe cao.
