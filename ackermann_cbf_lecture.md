# GIÁO TRÌNH TOÀN TẬP: ĐỘNG HỌC ACKERMANN KINEMATICS & CONTROL BARRIER FUNCTIONS (CBF-QP)

> **Chuyên Đề Lý Thuyết Điều Khiển Tự Động & Ứng Dụng Xe Tự Lái F1TENTH**  
> *Tác giả: Antigravity AI Team | Hệ Thống Xe Tự Lái F1TENTH ROS 2*

---

## 📌 CHƯƠNG 1: TỔNG QUAN HỆ THỐNG VÀ HỆ TỌA ĐỘ XE F1TENTH

### 1.1 Bài Toán Điều Khiển An Toàn Thời Gian Thực
Trong hệ thống điều khiển tự lái dựa trên Học máy (Imitation Learning / Deep RL), mô hình AI đóng vai trò làm bộ điều khiển thô (Nominal Controller) phát ra lệnh $u_{\text{nominal}} = [v_{\text{nom}}, \delta_{\text{nom}}]$. Tuy nhiên, AI không có chứng minh an toàn toán học và dễ phạm sai lầm (Out-of-Distribution) dẫn tới lao xe đâm vào tường ở tốc độ cao.

Bộ lọc **Control Barrier Function (CBF-QP)** đóng vai trò làm lá chắn an toàn (Safety Shield) can thiệp thời gian thực (real-time $< 1\text{ms}$). CBF sẽ nhận lệnh thô từ AI, tính toán bài toán tối ưu hóa lồi, và xuất ra lệnh an toàn $u_{\text{safe}} = [v_{\text{safe}}, \delta_{\text{safe}}]$ sao cho xe ít bị biến đổi nhất so với ý định gốc của AI nhưng **đảm bảo 100% không bao giờ đâm thủng rào chắn an toàn**.

### 1.2 Định Nghĩa Hệ Tọa Độ Thân Xe (Body Frame / `base_link`)
Xét xe F1TENTH chuyển động trong mặt phẳng 2D $(X, Y)$:
- **Gốc tọa độ $(0, 0)$:** Đặt tại tâm mắt cảm biến LiDAR phía đầu xe.
- **Trục X (Hướng dọc / Longitudinal Axis):** Hướng thẳng tiến phía trước theo chiều dọc thân xe.
- **Trục Y (Hướng ngang / Lateral Axis):** Hướng sang mạn bên trái xe vuông góc với trục X.
- **Góc Yaw ($\psi$):** Góc chệch hướng thân xe trong hệ tọa độ toàn cục.

```mermaid
graph TD
    A["Mặt tường / Vật cản P_i"] ---|r_i (Khoảng cách)| B["Tâm LiDAR (0,0)"]
    B -->|Trục X (Tiến)| C["Phía trước xe"]
    B -->|Trục Y (Sang trái)| D["Mạn trái xe"]
```

### 1.3 Chuyển Đổi Dữ Liệu LiDAR Sang Tọa Độ Descartes
Với mỗi tia LiDAR thứ $i$ đo được khoảng cách $r_i$ (m) ở góc tương đối $\phi_i$ (rad):
$$\begin{cases} x_i = r_i \cdot \cos(\phi_i) \\ y_i = r_i \cdot \sin(\phi_i) \end{cases}$$

Trong đó $\phi_i = 0$ ứng với tia chính giữa phía trước xe; $\phi_i > 0$ ứng với các tia bên trái; và $\phi_i < 0$ ứng với các tia bên phải.

---

## 📌 CHƯƠNG 2: MÔ HÌNH ĐỘNG HỌC XE ACKERMANN (ACKERMANN KINEMATICS)

### 2.1 Hình Học Bẻ Lái Ackermann
Khác với xe điều khiển vi sai (Unicycle / Differential Drive), xe F1TENTH bẻ lái bằng 2 bánh trước thông qua cơ cấu lái Ackermann với chiều dài cơ sở **$L = 0.33\text{ m}$** (khoảng cách từ trục bánh trước đến trục bánh sau).

```
             L (Chiều dài cơ sở = 0.33m)
        |<--------------------------------->|
       [Bánh Trước] ------------------- [Bánh Sau]
            \ (Góc lái delta)
```

### 2.2 Phương Trình Vi Phân Chuyển Động
Trạng thái hình học của xe được mô tả bởi hệ 3 phương trình vi phân:
$$\begin{cases} \frac{dx}{dt} = v \cdot \cos(\psi) \\ \frac{dy}{dt} = v \cdot \sin(\psi) \\ \frac{d\psi}{dt} = \frac{v}{L} \cdot \tan(\delta) \end{cases}$$

Trong đó:
- $v$: Vận tốc dài tiến của xe ($\text{m/s}$).
- $\delta$: Góc bẻ lái của 2 bánh trước ($\text{rad}$).
- $\dot{\psi} = \frac{d\psi}{dt}$: Tốc độ góc quay thân xe (Yaw Rate, $\text{rad/s}$).

> [!NOTE]
> **Ý nghĩa kỹ thuật:** Tốc độ xoay hướng của xe tỷ lệ thuận với vận tốc tiến $v$ và góc lái $\delta$. Khi xe chạy càng nhanh ($v$ lớn), cùng một góc lái $\delta$ sẽ làm xe đổi hướng cực kỳ nhanh!

### 2.3 Bán Kính Ôm Cua $R$
Khi bẻ lái góc $\delta$ cố định, xe di chuyển theo một đường tròn bán kính $R$:
$$R = \frac{L}{\tan(\delta)}$$

Gia tốc bám đường hướng tâm (Centripetal Acceleration) tác dụng lên bánh xe là:
$$a_{\text{lat}} = \frac{v^2}{R} = \frac{v^2 \cdot \tan|\delta|}{L}$$

---

## 📌 CHƯƠNG 3: QUÃNG ĐƯỜNG PHANH QUÁN TÍNH VẬT LÝ (DYNAMIC BRAKING CUSHION)

### 3.1 Nguyên Lý Vật Lý Phanh Biến Đổi Đều
Khi xe đang di chuyển ở vận tốc $v$ đâm về phía tường ở góc $\phi_i$, thành phần vận tốc lao thẳng vào tường là:
$$v_{\text{tường}} = v \cdot \cos(\phi_i)$$

Theo định luật chuyển động Newton: $v^2 - v_0^2 = 2 \cdot a \cdot S$. Khi hãm phanh với gia tốc hãm tối đa của động cơ $a_{\text{max\_brake}}$ ($\text{m/s}^2$), quãng đường trôi quán tính bắt buộc để xe giảm tốc từ $v_{\text{tường}}$ về $0\text{ m/s}$ là:

$$d_{\text{brake}}(v, \phi_i) = \frac{(v \cdot \cos\phi_i)^2}{2 \cdot a_{\text{max\_brake}}}$$

### 3.2 Bảng Phân Tích Mối Quan Hệ Phi Tuyến Giữa Vận Tốc Và Quãng Đường Phanh
Với gia tốc phanh mặc định $a_{\text{max\_brake}} = 3.5\text{ m/s}^2$:

| Vận tốc $v$ (m/s) | Vận tốc $v$ (km/h) | Quãng đường trôi phanh $d_{\text{brake}}$ (m) | Đánh giá rủi ro va chạm |
|---|---|---|---|
| **1.0 m/s** | 3.6 km/h | **0.14 m (14 cm)** | Rất an toàn, xe dừng ngay lập tức. |
| **2.0 m/s** | 7.2 km/h | **0.57 m (57 cm)** | Bắt đầu có độ trôi lớn, $d_{\text{min}}=0.2\text{m}$ sẽ bị đâm. |
| **3.0 m/s** | 10.8 km/h | **1.28 m (128 cm)** | Cần phanh cảnh báo từ khoảng cách xa 1.5m. |
| **5.0 m/s** | 18.0 km/h | **3.57 m (357 cm)** | Tốc độ rất cao, CBF phải phanh gấp từ xa 4m. |

---

## 📌 CHƯƠNG 4: THIẾT LẬP HÀM CONTROL BARRIER FUNCTION (CBF)

### 4.1 Hàm Barrier Động Học (Ackermann Barrier Function)
Định nghĩa hàm Barrier $h_i(x)$ đo "dung lượng an toàn còn lại" cho tia LiDAR thứ $i$:

$$h_i(x) = r_i - d_{\text{min}} - \frac{(v \cdot \cos\phi_i)^2}{2 \cdot a_{\text{max\_brake}}} \ge 0$$

- **$h_i(x) > 0$:** Xe nằm an toàn hoàn toàn bên trong Tập An Toàn (Safe Set $\mathcal{C}$).
- **$h_i(x) = 0$:** Xe nằm ngay trên biên an toàn ($\partial \mathcal{C}$).
- **$h_i(x) < 0$:** Nguy hiểm va chạm (Xe xâm nhập vùng cấm).

### 4.2 Bất Đẳng Thức CBF Invariance (Bảo Toàn An Toàn)
Điều kiện CBF bắt buộc tốc độ suy giảm của $h_i(x)$ phải thỏa mãn bất đẳng thức:
$$\frac{dh_i}{dt} + \gamma \cdot h_i(x) \ge -\epsilon$$

Khai triển đạo hàm $\frac{dh_i}{dt}$ theo Động học Ackermann tại vận tốc hiện tại $v_{\text{curr}}$:
$$g_v \cdot v + g_{\delta} \cdot \delta - \epsilon \le \gamma \cdot h_i(x)$$

Trong đó các hệ số ma trận đạo hàm được xác định bởi:
$$\begin{cases} g_v = \cos(\phi_i) \cdot \left[1 + \frac{v_{\text{curr}} \cdot \cos(\phi_i)}{a_{\text{max\_brake}}}\right] \\ g_{\delta} = -\frac{v_{\text{curr}}}{L} \cdot (r_i \cdot \sin\phi_i) \end{cases}$$

> [!IMPORTANT]
> **Phân tích ý nghĩa hình học của $g_{\delta}$:**  
> - Nếu vật cản ở bên trái xe ($\sin \phi_i > 0$) $\implies g_{\delta} < 0$. Bẻ lái sang phải ($\delta < 0$) làm vế trái $g_{\delta} \cdot \delta > 0$ bị triệt tiêu, giúp thỏa mãn bất đẳng thức an toàn.  
> - Tức là CBF tự động tính toán ra rằng **bẻ lái ngoặt sang phải sẽ giúp xe né được tường bên trái!**

---

## 📌 CHƯƠNG 5: BÀI TOÁN TỐI ƯU HÓA BẬC HAI (QUADRATIC PROGRAMMING - QP)

### 5.1 Mô Hình Tối Ưu Hóa Tốc Độ Cao Thời Gian Thực
Bộ lọc CBF thiết lập bài toán tối ưu bậc hai lồi 3 biến $x = [v, \delta, \epsilon]^T$:

$$\min_{v, \delta, \epsilon} \ \left[ \frac{1}{2} (v - v_{\text{nom}})^2 + \frac{5}{2} (\delta - \delta_{\text{nom}})^2 + \frac{p_{\text{slack}}}{2} \cdot \epsilon^2 \right]$$

Thỏa mãn hệ bất đẳng thức tuyến tính:
$$\begin{cases} G \cdot x \le h \\ \text{lb} \le x \le \text{ub} \end{cases}$$

Trong đó:
- Ma trận $G \in \mathbb{R}^{15 \times 3}$ gồm 15 hàng ứng với 15 tia LiDAR nguy hiểm nhất: $G_i = [g_v^{(i)}, g_{\delta}^{(i)}, -1.0]$.
- Vector $h \in \mathbb{R}^{15 \times 1}$: $h_i = \gamma \cdot \max(0.01, h_i(x))$.
- Trọng số Slack Variable $p_{\text{slack}} = 10,000$ để phạt cực nặng vi phạm vạch.
- Biên giới hạn vật lý: $\text{lb} = [0, -\delta_{\text{max}}, 0]$ và $\text{ub} = [v_{\text{max}}, \delta_{\text{max}}, 10.0]$.

---

## 📌 CHƯƠNG 6: 3 VÍ DỤ SỐ HỌC CHI TIẾT BƯỚC-THEO-BƯỚC

### 📌 VÍ DỤ 1: Xe Lao Thẳng Tường Ở Tốc Độ Thấp ($v = 1.0\text{ m/s}$)
- **Giả thiết:** $L = 0.33\text{m}, a_{\text{max\_brake}} = 3.5\text{ m/s}^2, d_{\text{min}} = 0.35\text{m}, \gamma = 2.0$.  
  AI đòi chạy thẳng: $v_{\text{nom}} = 1.0\text{ m/s}, \delta_{\text{nom}} = 0.0\text{ rad}$.  
  Tường phía trước: $r = 0.60\text{ m}, \phi = 0^\circ$.

- **Bước 1: Tính quãng đường phanh quán tính $d_{\text{brake}}$**  
  $d_{\text{brake}} = \frac{(1.0 \cdot \cos 0^\circ)^2}{2 \cdot 3.5} = \frac{1.0}{7.0} = 0.143\text{ m}$.

- **Bước 2: Tính hàm Barrier $h(x)$**  
  $h(x) = 0.60 - 0.35 - 0.143 = +0.107\text{ m} > 0$ (Xe vẫn AN TOÀN!).

- **KẾT QUẢ:** Solver QP trả về **$v_{\text{safe}} = 1.0\text{ m/s}, \delta_{\text{safe}} = 0.0\text{ rad}$**. CBF giữ nguyên lệnh AI 100% vì xe chưa gặp nguy hiểm.

---

### 📌 VÍ DỤ 2: Xe Lao Thẳng Tường Ở Tốc Độ Cao ($v = 3.0\text{ m/s}$)
- **Giả thiết:** Giống Ví dụ 1 nhưng AI đòi chạy rất nhanh: $v_{\text{nom}} = 3.0\text{ m/s}, \delta_{\text{nom}} = 0.0\text{ rad}$.  
  Tường phía trước: $r = 0.80\text{ m}, \phi = 0^\circ$.

- **Bước 1: Tính quãng đường phanh quán tính $d_{\text{brake}}$**  
  $d_{\text{brake}} = \frac{(3.0 \cdot 1.0)^2}{7.0} = \frac{9.0}{7.0} = 1.286\text{ m}$.

- **Bước 2: Tính hàm Barrier $h(x)$**  
  $h(x) = 0.80 - 0.35 - 1.286 = -0.836\text{ m} < 0$ (VI PHẠM NGUY HIỂM!).

- **Bước 3: Tính hệ số $g_v$**  
  $g_v = 1.0 \cdot \left[1 + \frac{3.0 \cdot 1.0}{3.5}\right] = 1 + 0.857 = 1.857$.

- **Bước 4: Bất đẳng thức CBF**  
  $1.857 \cdot v \le 2.0 \cdot (-0.836) = -1.672 \implies v \le 0.0\text{ m/s}$.

- **KẾT QUẢ:** CBF lập tức can thiệp hãm phanh **$v_{\text{safe}} = 0.0\text{ m/s}$**. Xe phanh dừng lại an toàn trước tường đúng mốc 0.35m thay vì bị trôi đâm vào tường!

---

### 📌 VÍ DỤ 3: Xe Lao Tường Nghiêng Bên Trái (Né Tường Bằng Động Học Ackermann)
- **Giả thiết:** $v_{\text{nom}} = 2.0\text{ m/s}, \delta_{\text{nom}} = 0.0\text{ rad}$.  
  Tường lệch bên trái: $r = 0.60\text{ m}, \phi = +15^\circ$ ($0.2588\text{ rad}$).

- **Bước 1: Tính $d_{\text{brake}}$ và $h(x)$**  
  $v_{\text{tường}} = 2.0 \cdot \cos 15^\circ = 1.932\text{ m/s}$.  
  $d_{\text{brake}} = \frac{1.932^2}{7.0} = 0.533\text{ m}$.  
  $h(x) = 0.60 - 0.35 - 0.533 = -0.283\text{ m} < 0$ (Nguy hiểm!).

- **Bước 2: Tính ma trận $g_v$ và $g_{\delta}$**  
  $g_v = \cos 15^\circ \cdot \left[1 + \frac{1.932}{3.5}\right] = 0.9659 \cdot 1.552 = 1.499$.  
  $g_{\delta} = -\frac{2.0}{0.33} \cdot (0.60 \cdot \sin 15^\circ) = -6.061 \cdot 0.1553 = -0.941$.

- **Bước 3: Bất đẳng thức CBF**  
  $1.499 \cdot v - 0.941 \cdot \delta \le 2.0 \cdot (-0.283) = -0.566$.

- **Bước 4: Giải tối ưu QP**  
  Solver chọn bẻ góc lái né sang phải: $\delta_{\text{safe}} = -0.30\text{ rad}$ ($-17.2^\circ$).  
  $1.499 \cdot v + 0.282 \le -0.566 \implies 1.499 \cdot v \le -0.848 \implies v_{\text{safe}} \approx 0.85\text{ m/s}$.

- **KẾT QUẢ:** Xe không bị dừng khựng ngắt ngắt, mà **vừa giảm tốc nhẹ xuống 0.85 m/s vừa bẻ ngoặt lái $-17.2^\circ$ sang phải** để ôm cua né tường mượt mà!

---

## 📌 CHƯƠNG 7: TỔNG KẾT BẢNG THAM SỐ VÀ HƯỚNG DẪN TUNING

| Tham số | Tên trong code | Mặc định | Khoảng khuyên dùng | Hướng dẫn Tinh chỉnh (Tuning Guide) |
|---|---|---|---|---|
| **Khoảng cách tĩnh** | `d_min` | `0.35` m | 0.25m – 0.40m | Tăng lên 0.4m nếu chạy không gian rộng; giảm xuống 0.28m nếu muốn xe đua ép sát tường hơn. |
| **Gia tốc phanh** | `a_max_brake` | `3.5` m/s² | 2.5 – 5.0 m/s² | Giảm xuống 2.5 nếu xe phanh kém/trượt lốp để CBF phanh từ xa hơn. |
| **Chiều dài gầm** | `wheelbase` | `0.33` m | 0.32m – 0.34m | Cố định theo kích thước vật lý gầm xe F1TENTH. |
| **CBF Gain** | `gamma` | `2.0` | 1.5 – 4.5 | Tăng lên 4.0 nếu muốn CBF phản ứng phanh/né gắt hơn khi sát tường. |
| **Vận tốc trần** | `v_max` | `3.0` m/s | 2.0m/s – 7.0m/s | Giới hạn vận tốc an toàn tối đa cho phép xe chạy. |
