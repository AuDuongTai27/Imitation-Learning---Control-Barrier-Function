# DÀN Ý BÀI BÁO CÁO KHOA HỌC / SCIENTIFIC PAPER OUTLINE (CONFIRMED SETUP)

**Tên đề tài gợi ý (Suggested Titles):**
1. *Sim-to-Real Transfer of Imitation Learning with Control Barrier Functions for Autonomous Racing on the F1TENTH Platform*
2. *Application of Mixed-Domain Imitation Learning and Safety-Critical CBF for Autonomous Navigation on F1TENTH ROS 2*
3. *A Practical Sim-to-Real Framework Combining DAgger-based Imitation Learning and Real-Time CBF-QP Safety Filtering for F1TENTH Vehicles*

---

## 1. ABSTRACT (Tóm tắt bài báo)

* **Context & Motivation (Bối cảnh):** Sự phát triển của các phương pháp học máy (Imitation Learning) trong điều khiển xe tự hành F1TENTH và thách thức về khoảng cách giữa mô phỏng và thực tế (Sim-to-Real gap) cùng vấn đề đảm bảo an toàn phần cứng.
* **Problem (Vấn đề):** Các chính sách học bắt chước (IL) thuần túy huấn luyện trong mô phỏng thường dễ mất an toàn khi triển khai thực tế do nhiễu cảm biến, động lực học thực tế khác biệt và thiếu cơ chế cam kết an toàn thời gian thực.
* **Proposed Approach (Giải pháp đề xuất):** 
  * Xây dựng hệ thống chuyên gia kết hợp **Pure Pursuit + RRT** để thu thập dữ liệu di chuyển bám đường và né vật cản.
  * Mạng **MLP (Multi-Layer Perceptron)** huấn luyện bằng phương pháp **Co-training** trên tập dữ liệu hỗn hợp gồm **20,000 mẫu mô phỏng (Sim)** và **6,000 mẫu thực tế (Real)**. Mạng nhận đầu vào là LaserScan thu gọn (108–180 tia) + Vận tốc hiện tại $(v, \omega)$.
  * Tích hợp bộ lọc an toàn thời gian thực giải bằng **Quadratic Programming (CBF-QP Solver - OSQP/CVXOPT)** trên ROS 2 để giới hạn tốc độ và góc lái an toàn tối ưu.
* **Key Findings & Results (Kết quả chính):** Xe F1TENTH chạy thực tế bám đường ổn định, giảm thiểu Sim-to-Real gap nhờ co-training, và bộ lọc CBF-QP đảm bảo 0% sự cố va chạm ngay cả khi mạng IL đưa ra lệnh vượt ngưỡng an toàn.

---

## 2. SECTION I: INTRODUCTION (Mở đầu)

### 1.1. Bối cảnh & Lý do chọn đề tài (Background & Motivation)
* Giới thiệu về nền tảng đua xe tự hành tỉ lệ 1/10 (**F1TENTH**) và hệ điều hành **ROS 2**.
* Tầm quan trọng của việc ứng dụng kỹ thuật Học bắt chước (**Imitation Learning - IL**) trong điều khiển phản hồi nhanh (reactive control).

### 1.2. Thách thức nghiên cứu (Problem Statement & Challenges)
* **Khoảng cách Sim-to-Real (Sim-to-Real Gap):** Sự sai lệch giữa môi trường mô phỏng (Gazebo/F1TENTH Gym) và xe thực tế (ma sát mặt đường, nhiễu LiDAR, độ trễ động cơ).
* **Rủi ro va chạm & An toàn phần cứng:** Mô hình IL thuần túy là "hộp đen" (black-box), không có cam kết an toàn toán học, dễ gây đâm đụng hư hỏng xe khi gặp trường hợp ngoài phân phối dữ liệu (Out-of-Distribution - OOD).

### 1.3. Định hướng bài báo (Scope & Focus)
* **Tập trung vào ứng dụng và thử nghiệm thực tế (Application & Empirical Study):** Bài báo tập trung vào giải pháp tích hợp hệ thống, quy trình thu thập dữ liệu kết hợp và triển khai thực nghiệm trên xe F1TENTH thực tế chạy ROS 2, không tập trung xây dựng thuật toán lý thuyết mới.

### 1.4. Đóng góp chính của bài báo (Main Contributions)
1. Triển khai thành công chu trình thu thập dữ liệu chuyên gia (**Pure Pursuit + RRT**) cho xe F1TENTH bám quỹ đạo và né vật cản.
2. Xây dựng quy trình **Co-training** huấn luyện mạng MLP trên tập dữ liệu kết hợp (20,000 Sim + 6,000 Real) giúp thu hẹp Sim-to-Real gap.
3. Đóng gói mô hình IL kết hợp bộ lọc tối ưu **CBF-QP Solver** thời gian thực trên kiến trúc ROS 2, chứng minh tính khả thi và độ an toàn phần cứng F1TENTH.

---

## 3. SECTION II: RELATED WORK (Tổng quan nghiên cứu)

### 2.1. Autonomous Navigation & Racing on F1TENTH Platform
* Các phương pháp điều khiển truyền thống (Pure Pursuit, MPC) và lập kế hoạch (RRT, RRT*).

### 2.2. Imitation Learning & Sim-to-Real Transfer in Robotics
* **Imitation Learning:** Behavioral Cloning (BC), DAgger, MPD.
* **Sim-to-Real techniques:** Co-training / Mixed dataset training trong điều khiển xe robot.

### 2.3. Control Barrier Functions (CBF) for Safety-Critical Control
* Khái niệm bộ lọc an toàn thời gian thực (Safety Filter / Corrective Controller via QP).

---

## 4. SECTION III: METHODOLOGY & SYSTEM ARCHITECTURE (Phương pháp luận & Kiến trúc)

```
[LiDAR Scan (Downsampled 108-180 rays) + Speed (v, omega)] 
        │
        ▼
[Imitation Learning Policy: MLP] ───(Nominal Control: u_IL = [v_cmd, steer_cmd])───► [CBF-QP Solver Node]
                                                                                           │
                                                                           (Safe Control: u_safe = [v_safe, steer_safe])
                                                                                           │
                                                                                           ▼
                                                                                   [F1TENTH Actuators]
```

### 3.1. Tổng quan kiến trúc hệ thống (System Overview)
* Trình bày sơ đồ khối hệ thống chạy trên nền tảng ROS 2.

### 3.2. Thu thập dữ liệu Chuyên gia (Expert Planner: Pure Pursuit + RRT)
* **Pure Pursuit:** Bám đường đua / quỹ đạo tham chiếu.
* **RRT (Rapidly-exploring Random Tree):** Lập kế hoạch cục bộ tránh chướng ngại vật khi phát hiện vật cản.

### 3.3. Chiến lược Dữ liệu & Co-training (Mixed-Domain Co-training Strategy)
* **Dữ liệu Mô phỏng (Sim Data - 20,000 samples):** Thu thập trong simulator với nhiều cấu hình đường đua.
* **Dữ liệu Thực tế (Real Data - 6,000 samples):** Thu thập trực tiếp trên xe F1TENTH để chụp lại nhiễu cảm biến và ma sát thực tế.
* **Co-training Pipeline:** Trộn ngẫu nhiên (shuffle joint training) dữ liệu Sim + Real để huấn luyện weights cho mạng MLP ngay từ đầu.

### 3.4. Kiến trúc Mạng Nơ-ron Imitation Learning (MLP Architecture)
* **Đầu vào (Input Vector):** 
  * 108 – 180 tia LiDAR quét đã thu gọn (Downsampled LaserScan).
  * Vận tốc hiện tại của xe $v$ và vận tốc góc $\omega$.
* **Cấu trúc Mạng:** Multi-Layer Perceptron (MLP) với các lớp Fully-Connected + ReLU / GELU.
* **Đầu ra (Output Vector):** $u_{IL} = [v_{cmd}, \delta_{cmd}]$ (Tốc độ mục tiêu & Góc lái mục tiêu).

### 3.5. Bộ lọc an toàn thời gian thực CBF-QP (Real-Time CBF-QP Solver)
* **Mô hình động học xe (Kinematic Bicycle Model):**
  $$\dot{x} = v \cos\theta, \quad \dot{y} = v \sin\theta, \quad \dot{\theta} = \frac{v}{L} \tan\delta$$
* **Thiết lập tập an toàn (Safe Set):**
  $$h(x) = d_{obs} - d_{min} \ge 0$$
* **Bài toán Tối ưu hóa Quadratic Programming (QP):**
  $$\min_{u_{safe}} \| u_{safe} - u_{IL} \|^2 \quad \text{s.t.} \quad \nabla h(x) \cdot f(x, u_{safe}) + \gamma h(x) \ge 0, \quad u_{min} \le u_{safe} \le u_{max}$$
  *(Giải bằng solver OSQP/CVXOPT thời gian thực trên node ROS 2).*

---

## 5. SECTION IV: EXPERIMENTAL SETUP & IMPLEMENTATION (Triển khai & Thử nghiệm)

### 4.1. Cấu hình Phần cứng & Phần mềm xe F1TENTH
* Xe F1TENTH 1/10, VESC Controller, Hokuyo/RPLiDAR, Jetson Orin / NUC.
* ROS 2 Humble/Galactic, PyTorch / ONNX Runtime, OSQP / CVXOPT Python solver.

### 4.2. Các kịch bản thử nghiệm (Experimental Scenarios)
1. **Scenario 1: Sim-to-Real Baseline Comparison**
   * Baseline A: Model chỉ train bằng 100% Sim Data (20,000 samples).
   * Proposed: Model Co-training bằng Mixed Data (20,000 Sim + 6,000 Real).
2. **Scenario 2: Safety Verification (MLP Pure vs. MLP + CBF-QP Filter)**
   * Đặt vật cản bất ngờ để quan sát phản ứng can thiệp của bài toán tối ưu QP.

---

## 6. SECTION V: RESULTS AND DISCUSSION (Kết quả & Thảo luận)

### 5.1. Hiệu quả của Co-training (Mixed Dataset)
* Bảng so sánh tỉ lệ hoàn thành vòng đua và sai số bám làn (CTE) giữa *Sim-only* và *Mixed Co-training*.

### 5.2. Hiệu quả của Bộ lọc CBF-QP
* Biểu đồ chuỗi thời gian (Time-series plot): So sánh $u_{IL}$ đề xuất từ mạng MLP và $u_{safe}$ đã được điều chỉnh qua solver QP.

### 5.3. Thời gian suy luận & Tính thời gian thực (Real-time Latency)
* Tần số chạy của Node ROS 2 (MLP Inference ~2ms + CBF-QP Solver ~3ms) $\rightarrow$ Đạt tần số điều khiển > 50 Hz.

---

## 7. SECTION VI: CONCLUSION AND FUTURE WORK (Kết luận & Hướng phát triển)

### 7.1. Kết luận
* Khẳng định tính hiệu quả và độ an toàn của chu trình tích hợp **Pure Pursuit + RRT**, **Co-training Mixed Sim-to-Real**, và **CBF-QP Safety Shield** trên xe F1TENTH ROS 2.

### 7.2. Hạn chế & Hướng phát triển
* Mới áp dụng CBF cơ bản, tương lai có thể mở rộng sang Adaptive CBF.
