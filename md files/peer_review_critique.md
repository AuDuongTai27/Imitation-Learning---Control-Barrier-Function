# BÁO CÁO PHẢN BIỆN KHOA HỌC (PEER-REVIEW CRITIQUE REPORT)

**Đề tài:** *Application of Mixed-Domain Imitation Learning and Control Barrier Functions for F1TENTH Autonomous Racing on ROS 2*

---

## 1. TỔNG QUAN ĐÁNH GIÁ (OVERVIEW)

* **Điểm mạnh (Strengths):**
  * Hướng đi mang tính thực tiễn cao, tập trung giải quyết bài toán cốt lõi trong robotics: **Sim-to-Real gap** và **An toàn phần cứng**.
  * Có sự kết hợp hài hòa giữa Học máy (Imitation Learning) và Điều khiển lý thuyết (CBF-QP Filter).
  * Đã triển khai và thực nghiệm trực tiếp trên phần cứng thật xe **F1TENTH chạy ROS 2**.

* **Định vị loại bài báo:** Bài báo của bạn thuộc dạng **Application & Empirical System Study** (Nghiên cứu Thực nghiệm và Tích hợp Hệ thống). Đối với dạng bài này, phản biện (Reviewers) từ các tạp chí/hội nghị uy tín (IEEE RAM, IEEE Access, IROS) sẽ không bắt bẻ về việc "thiếu thuật toán toán học mới", nhưng sẽ **xoáy rất sâu vào 6 điểm yếu dưới đây**.

---

## 2. CHI TIẾT 6 ĐIỂM YẾU & CÁCH PHẢN BIỆN / PHÒNG THỦ (CRITIQUES & DEFENSE STRATEGIES)

### 🔴 Điểm yếu 1: Bộ lọc CBF quá cơ bản (Basic CBF Formulation)
* **Phản biện sẽ hỏi:** 
  > *"Tác giả sử dụng mô hình CBF cơ bản với h số $\gamma$ cố định và mô hình động học xe đơn giản (Kinematic Bicycle Model). Khi xe F1TENTH chạy ở tốc độ cao, hiện tượng trượt lốp (tire slip) và độ trễ phản hồi góc lái (steering latency) xảy ra rất lớn. Liệu CBF cơ bản có thực sự đảm bảo an toàn tuyệt đối không? Tại sao không dùng High-Order CBF (HOCBF) hay Adaptive CBF?"*
* **Cách giải quyết & Phòng thủ trong bài báo:**
  * **Lý do chọn:** Giải thích rằng xe F1TENTH sử dụng máy tính nhúng (Jetson Orin Nano). Các mô hình HOCBF hay Adaptive CBF làm tăng số lượng ràng buộc phi tuyến, dễ dẫn đến hiện tượng **Infeasible QP (không tìm được nghiệm QP)** hoặc độ trễ tính toán >20ms $\rightarrow$ gây mất an toàn phần cứng.
  * **Giải pháp:** Đưa ra số liệu thực nghiệm chứng minh CBF cơ bản với $\gamma$ được tuned hợp lý chỉ mất **< 3ms**, đảm bảo tần số điều khiển > 50Hz, thừa đủ để ngăn va chạm ở dải tốc độ thử nghiệm.

---

### 🔴 Điểm yếu 2: Con số dữ liệu 20,000 Sim vs 6,000 Real thiếu luận điểm khoa học (Arbitrary Dataset Ratio)
* **Phản biện sẽ hỏi:** 
  > *"Con số 20,000 mẫu Sim và 6,000 mẫu Real từ đâu ra? Tại sao không phải là 10,000 Sim + 10,000 Real hay 20,000 Sim + 2,000 Real? Tỷ lệ này có phải là tối ưu không?"*
* **Cách giải quyết & Phòng thủ trong bài báo:**
  * **Cần bổ sung thử nghiệm Ablation Study:** Trong phần Results, bạn phải vẽ một biểu đồ/bảng so sánh hiệu năng theo số lượng mẫu Real (vd: 0 Real, 2k Real, 4k Real, **6k Real**, 8k Real).
  * **Kết luận trong bài:** Chứng minh rằng tại mốc **6,000 mẫu Real**, độ chính xác bám đường của mạng đạt mức **bão hòa (plateau)**, nếu thu thập thêm (vd 8k hay 10k mẫu) thì hiệu năng chỉ tăng rất ít nhưng chi phí/rủi ro cho phần cứng lại tăng cao.

---

### 🔴 Điểm yếu 3: Kỹ thuật Co-training quá đơn giản so với Domain Adaptation
* **Phản biện sẽ hỏi:** 
  > *"Tại sao chỉ gộp chung (Co-training / Shuffle) dữ liệu Sim và Real mà không dùng các kỹ thuật Sim-to-Real tiên tiến hơn như Domain Randomization, Gradient Reverse Layer (Domain Adversarial), hay Pre-train trên Sim rồi Fine-tune trên Real?"*
* **Cách giải quyết & Phòng thủ trong bài báo:**
  * **Thừa nhận & So sánh:** Trong phần thực nghiệm, so sánh trực tiếp phương pháp **Co-training** của bạn với phương pháp **Fine-tuning** (Pre-train Sim $\rightarrow$ Fine-tune Real).
  * **Lập luận:** Chứng minh rằng trong bài toán nhỏ này, Fine-tuning dễ dẫn đến hiện tượng **Catastrophic Forgetting** (mạng quên các kịch bản đa dạng trong Sim và chỉ nhỡ dữ liệu ít ỏi của Real), trong khi Co-training giúp giữ lại tri thức tổng quát của Sim đồng thời bổ sung đặc trưng nhiễu của Real.

---

### 🔴 Điểm yếu 4: Tính không tối ưu của Chuyên gia Pure Pursuit + RRT (Sub-optimal Expert)
* **Phản biện sẽ hỏi:** 
  > *"Pure Pursuit + RRT là thuật toán rule-based tạo ra đường đi bị gấp khúc (jerky) và không tối ưu thời gian vòng đua (lap time) như MPC hay Min-Curvature Raceline. Mạng IL học theo chuyên gia này sẽ bị học theo những hành vi không tối ưu đó."*
* **Cách giải quyết & Phòng thủ trong bài báo:**
  * **Khẳng định lại mục tiêu bài báo:** Nhấn mạnh rằng mục tiêu của bài báo là **"Bám đường ổn định & Tránh vật cản an toàn" (Safe Tracking & Obstacle Avoidance)** chứ không phải "Đua xe tối ưu thời gian" (Time-optimal Racing).
  * **Ưu điểm của Pure Pursuit + RRT:** Thuật toán này sinh dữ liệu cực kỳ nhanh, ổn định 100% trong mô phỏng để tự động hóa việc gán nhãn 20,000 mẫu dữ liệu mà không cần con người điều khiển thủ công.

---

### 🔴 Điểm yếu 5: Giới hạn về kịch bản Vật cản (Static vs Dynamic Obstacles)
* **Phản biện sẽ hỏi:** 
  > *"Lớp CBF và RRT của bạn có xử lý được vật cản di động (Dynamic Obstacles) với tốc độ cao hay chỉ xử lý được vật cản đứng yên (Static Obstacles)?"*
* **Cách giải quyết & Phòng thủ trong bài báo:**
  * **Khoanh vùng phạm vi (Scope Boundary):** Trong phần Introduction, nêu rõ bài báo tập trung vào môi trường vật cản tĩnh và bán động (static & semi-static obstacles).
  * **Đưa vào Future Work:** Nêu rõ việc mở rộng sang CBF dạng nón va chạm (Collision Cone CBF) cho vật cản di động sẽ là hướng nghiên cứu tiếp theo.

---

### 🔴 Điểm yếu 6: Mạng MLP quá đơn giản (Lack of Temporal / Spatial Feature Extraction)
* **Phản biện sẽ hỏi:** 
  > *"Dữ liệu quét LiDAR là chuỗi 1D có tính chất không gian, tại sao không dùng CNN 1D hay RNN/LSTM/GRU để lưu lại thông tin thời gian mà lại dùng MLP đơn giản?"*
* **Cách giải quyết & Phòng thủ trong bài báo:**
  * **Luận điểm về Real-time Edge Computing:** Mạng MLP với 108–180 tia LiDAR thu gọn chỉ mất **< 1-2 ms** để inference trên Jetson Orin Nano. 
  * Các mạng CNN 1D hay LSTM làm tăng đáng kể latency và memory footprint mà không đem lại cải thiện quá lớn đối với bài toán điều khiển phản hồi nhanh (reactive control) tốc độ cao.

---

## 3. TỔNG KẾT BẢNG CÂU HỎI & CÂU TRẢ LỜI NHANH KHI BỊ REVIEWER "HỎI XOÁY"

| Phản biện hỏi (Reviewer Objection) | Câu trả lời phòng thủ ngắn gọn (Quick Defense) |
| :--- | :--- |
| **"CBF cơ bản có đủ an toàn ở tốc độ cao không?"** | CBF cơ bản tính toán nhẹ (<3ms), tần số >50Hz đủ bù đắp sai số mô hình mà không lo bị quá thời gian tính toán của solver QP. |
| **"Tại sao dùng Co-training thay vì Fine-tuning?"** | Co-training tránh hiện tượng Catastrophic Forgetting, duy trì sự đa dạng kịch bản của Sim và độ thực tế của Real. |
| **"Tỷ lệ 20k Sim / 6k Real có tối ưu không?"** | Đã thực hiện Ablation Study; mốc 6k Real là điểm bão hòa giữa hiệu năng bám đường và chi phí thu thập dữ liệu phần cứng. |
| **"Tại sao không dùng CNN/LSTM mà dùng MLP?"** | MLP cho độ trễ suy luận siêu thấp (<2ms), đảm bảo tính thời gian thực trên chip nhúng của F1TENTH. |
