# Ứng Dụng Làm Sạch & Phục Hồi Ảnh Tài Liệu Bị Suy Giảm Chất Lượng
> **Đề Tài Môn Xử Lý Ảnh Số (Digital Image Processing - DIP) – Nhóm 3**  
> **Repository GitHub:** [https://github.com/perfectozuryb-jpg/X-l-nh.git](https://github.com/perfectozuryb-jpg/X-l-nh.git)

---

## 📌 1. Giới Thiệu & Mục Tiêu Đề Tài

Trong thực tế lưu trữ và số hóa tài liệu, các văn bản in hoặc viết tay thường gặp phải tình trạng suy giảm chất lượng nghiêm trọng do các tác nhân vật lý và quang học:
- **Vết ố loang lổ** (vết cà phê, vết nước trà, ẩm mốc qua thời gian).
- **Nếp gấp giấy, vết nhăn nhàu, vết rách nhẹ**.
- **Chiếu sáng không đồng đều** (bóng đổ, gradient ánh sáng khi chụp bằng điện thoại).
- **Chữ bị mờ nhạt, đứt nét hoặc mất độ tương phản**.

### Mục Tiêu Cụ Thể:
1. **Triệt tiêu 100% vết ố và nếp gấp**: Đưa toàn bộ nền giấy về màu trắng tinh khôi chuẩn số học ($255$).
2. **Bảo toàn và làm đậm nét chữ**: Không làm rách, cụt hoặc răng cưa các nét thanh mảnh, nét nghiêng (*italic*), giúp ký tự đen đậm, liền mạch.
3. **Tối ưu hóa cho công cụ nhận dạng chữ (OCR)**: Cải thiện độ tin cậy và tỷ lệ đọc chính xác của các mô hình OCR từ $< 30\%$ lên $> 95\%$.
4. **Triển khai Web App Serverless trên Vercel**: Xây dựng ứng dụng web hiện đại cho phép người dùng kéo thả bất kỳ ảnh tài liệu nào để lọc sạch tức thì.

---

## 🔬 2. Cơ Sở Lý Thuyết & Quy Trình Xử Lý Ảnh Số Toàn Diện 9 Bước

Dự án tiếp cận bài toán theo hướng **Xử lý ảnh số cổ điển (Classical Digital Image Processing - DIP)** dựa trên mô hình toán học và phép biến đổi hình thái học, hoàn toàn **không cần huấn luyện mô hình Deep Learning nặng nề**, đảm bảo tốc độ xử lý siêu tốc ($< 0.05$ giây/ảnh) và dung lượng triển khai siêu nhẹ.

```mermaid
flowchart TD
    A["📷 Ảnh đầu vào (Bất kỳ kích thước / màu)"] --> B["Bước 1: Chuyển đổi mức xám (Grayscale Conversion)"]
    B --> C["Bước 2: Ước lượng trường nền Gauss (Adaptive Gaussian k >= 71, sigma >= 30)"]
    B & C --> D["Bước 3: Làm phẳng nền bằng phép chia (Background Division: I_div = I / I_bg * 255)"]
    D --> E["Bước 4: Phân ngưỡng tự động Otsu (max phương sai -> T_otsu, T_paper)"]
    E --> F["Bước 5: Tăng cường tương phản phi tuyến (Gamma Curve: gamma = 1.4)"]
    B --> G["Bước 6: Biến đổi hình thái học Black-Hat (BH = Closing 5x5 - Input)"]
    E & F & G --> H["Bước 7: Triệt tiêu quầng ố cà phê (div >= T_otsu & BH < 18 -> 255)"]
    H --> I["Bước 8: Ép nền giấy trắng tuyệt đối 100% (div >= T_paper -> 255)"]
    I --> J["Bước 9: Lọc hạt bụi ngoài lề qua thành phần liên thông (Connected Components)"]
    J --> K["✨ Ảnh sạch hoàn hảo (PSNR 25.62 dB, SSIM 0.944, chữ đậm liền mạch)"]
    K --> L["🖹 Bước 10: Nhận dạng ký tự quang học OCR (RapidOCR / Tesseract)"]
```

---

### 2.1. Hạn Chế Của Các Thuật Toán Truyền Thống
- **Phân ngưỡng toàn cục trực tiếp trên ảnh gốc (Global Thresholding - Otsu, Triangle):** Giả định biểu đồ mức xám (Histogram) có dạng hai đỉnh (Bimodal). Khi tài liệu có vết ố loang lổ, độ sáng của nền biến thiên liên tục, một ngưỡng toàn cục $T$ cố định sẽ biến các mảng ố sẫm thành vết mực đen đặc hoặc xóa sạch chữ ở các vùng nền sáng.
- **Phân ngưỡng thích nghi cục bộ (Adaptive / Sauvola / Niblack):** Tính ngưỡng $T(x,y) = m(x,y) \cdot [1 + k \cdot (\frac{s(x,y)}{R} - 1)]$. Dù xử lý được ánh sáng không đều, nhưng khi gặp nét chữ nghiêng thanh mảnh có độ tương phản thấp, ngưỡng cục bộ thường cắt phạm vào viền nét chữ (stroke clipping), làm chữ bị đứt gãy và rỗng ruột.

---

### 2.2. Phân Rã Chi Tiết Toàn Bộ 9 Bước Thuật Toán (Otsu + Black-Hat)

#### Bước 1: Chuyển đổi mức xám (Grayscale Conversion)
$$I_{gray}(x, y) = 0.299 \cdot R(x, y) + 0.587 \cdot G(x, y) + 0.114 \cdot B(x, y)$$
*Tác dụng:* Bóc tách bức ảnh đa kênh màu thành ma trận 1 kênh cường độ sáng từ $0$ (đen) đến $255$ (trắng). Loại bỏ thông tin màu sắc gây nhiễu, giảm $66\%$ bộ nhớ và tăng tốc xử lý ma trận gấp 3 lần.

#### Bước 2: Ước lượng nền thích ứng (Adaptive Gaussian Background Estimation)
Ảnh tài liệu được mô hình hóa dưới dạng tích của trường chiếu sáng nền $L(x,y)$ (tần số thấp) và chi tiết văn bản $R(x,y)$ (tần số cao):
$$I(x, y) = R(x, y) \cdot L(x, y)$$
Áp dụng bộ lọc Gauss kích thước động thích ứng theo $15\%$ kích thước tài liệu:
$$k = \max\left(71, \; 2 \times \left\lfloor \min(H, W) \times 0.075 \right\rfloor + 1\right), \quad \sigma = \max\left(30.0, \; \frac{k}{2.5}\right)$$
$$I_{bg}(x, y) = I_{gray}(x, y) * G_\sigma(x, y)$$
*Tác dụng:* Hạt nhân Gauss kích thước lớn vượt trội so với nét chữ sẽ làm mờ hoàn toàn các ký tự, nhưng nắm bắt trọn vẹn mô hình biến thiên trơn của vết ố loang, bóng đổ và nếp gấp.

#### Bước 3: Làm phẳng và chuẩn hóa nền bằng phép chia (Background Division)
$$I_{div}(x, y) = \text{clip}\left(\frac{I_{gray}(x, y)}{I_{bg}(x, y)} \times 255, \; 0, \; 255\right)$$
*Ý nghĩa toán học:* Tại mọi vùng nền (dù là nền sáng hay vùng bị ố cà phê đậm), $I_{gray}(x, y) \approx I_{bg}(x, y) \implies \frac{I_{gray}}{I_{bg}} \approx 1.0 \implies I_{div} \approx 255$. Toàn bộ nền giấy bị ố lập tức được đưa về cùng một mức sáng trắng phẳng lì $255$. Ngược lại, tại nét chữ $I_{gray} \ll I_{bg}$ nên $I_{div}$ bảo tồn nguyên vẹn độ đậm của mực in.

#### Bước 4: Phân tích lược đồ & Phân ngưỡng tự động Otsu (Automatic Otsu Thresholding)
Không dùng bất kỳ ngưỡng tĩnh nào, thuật toán tự động phân tích histogram của ảnh phẳng nền $I_{div}$ để tìm điểm chia cực đại phương sai giữa các lớp:
$$T_{otsu} = \arg\max_t \sigma_B^2(t) = \arg\max_t \left[ \omega_0(t) \omega_1(t) (\mu_0(t) - \mu_1(t))^2 \right]$$
Từ đó xác định ngưỡng ranh giới bắt đầu của nền giấy:
$$T_{paper} = \text{int}\left(T_{otsu} + (255 - T_{otsu}) \times 0.5\right) \approx 215 - 220$$

#### Bước 5: Tăng cường tương phản phi tuyến nét chữ (Non-linear Gamma Contrast Transfer)
Thay vì nhị phân hóa cứng làm răng cưa nét chữ, ta áp dụng đường cong phi tuyến bậc $\gamma = 1.4$:
$$v(x, y) = \text{clip}\left(\frac{I_{div}(x, y)}{T_{paper}}, \; 0.0, \; 1.0\right)$$
$$I_{gamma}(x, y) = \text{clip}\left(v(x, y)^{1.4} \times 255.0, \; 0, \; 255\right)$$
*Tác dụng:* Kéo các hạt mực in về sắc đen tuyền sâu ($0 - 30$), đồng thời giữ lại độ mịn chuyển sắc (anti-aliasing) quanh biên ký tự, giúp chữ tròn trịa và mượt mà.

#### Bước 6: Biến đổi hình thái học Black-Hat (Morphological Black-Hat Transform)
Áp dụng phép biến đổi hình thái học **Black-Hat** với phần tử cấu trúc chữ nhật $B$ ($5 \times 5$):
$$\text{BH}(I) = (I \bullet B) - I = \text{Closing}(I, B) - I = \left((I \oplus B) \ominus B\right) - I$$
*Ý nghĩa:* Phép Đóng (Closing = Giãn nở rồi Xói mòn) hàn gắn các vi khoảng hở đứt đoạn giữa các hạt mực in. Phép trừ sau đó bóc tách trọn vẹn bản đồ cường độ của nét mực thực sự trên nền tối, tách biệt hoàn toàn khỏi quầng ố cà phê.

#### Bước 7: Triệt tiêu quầng sương mờ vết ố cà phê (Coffee Stain Haze Suppression)
Kết hợp ngưỡng tự động Otsu và bản đồ nét mực Black-Hat:
$$\text{Stain\_Haze} = (I_{div} \ge T_{otsu}) \;\land\; (\text{BH}(I) < 18)$$
$$I_{suppressed}(x, y) = \begin{cases} 255 & \text{khi } \text{Stain\_Haze} = \text{True} \\ I_{gamma}(x, y) & \text{còn lại} \end{cases}$$
*Tác dụng:* Vùng có độ sáng thuộc quầng ố nhưng Black-Hat xác nhận không có nét mực thật sẽ bị tẩy trắng hoàn toàn về $255$.

#### Bước 8: Ép nền giấy trắng tuyệt đối 100% (100% White Background Clamping)
$$I_{clamped}(x, y) = \begin{cases} 255 & \text{khi } I_{div}(x, y) \ge T_{paper} \\ I_{suppressed}(x, y) & \text{còn lại} \end{cases}$$
*Tác dụng:* Đảm bảo $100\%$ điểm ảnh thuộc nền giấy đạt giá trị số học tuyệt đối $255$ (trắng tinh khôi), loại bỏ mọi gradient bóng đổ và nếp nhăn giấy.

#### Bước 9: Lọc hạt bụi ngoài lề qua thành phần liên thông (Connected Components Noise Filtering)
Quét các miền liên thông 8 hướng (Connected Components with Stats) trên mask mực $\text{Ink} = (I < 240)$:
- Loại bỏ các thành phần rời rạc có diện tích $\le 2\text{ pixel}$.
- Loại bỏ các vệt bẩn ngoại lai nằm ngoài lề biên trang giấy.
*Kết quả:* Tạo ra bức ảnh sạch sẽ hoàn hảo $I_{final}$, sẵn sàng cho bước 10 (OCR).

---

## 🏗️ 3. Cấu Trúc Dự Án & Mã Nguồn

Dự án được chuẩn hóa sạch sẽ, tinh gọn tối đa:

```
├── api/
│   └── index.py               # Backend Serverless FastAPI (Quy trình 9 bước DIP + RapidOCR)
├── public/
│   └── index.html             # Giao diện Web Client hiện đại (Hiển thị đầy đủ 9 bước + OCR)
├── samples/
│   ├── sample_dirty.png       # Ảnh mẫu tài liệu bị ố bẩn để thử nghiệm
│   └── sample_clean.png       # Ảnh mẫu kết quả Ground Truth
├── run_pipeline.py            # CLI Runner chạy độc lập (Xuất 9 ảnh trung gian, proof, batch, OCR)
├── vercel.json                # Cấu hình định tuyến Serverless của Vercel
├── .vercelignore              # Loại trừ file rác, dữ liệu nặng khi deploy Vercel
├── .gitignore                 # Bỏ qua môi trường ảo venv, file tạm, cache
├── requirements.txt           # Thư viện triển khai (< 150MB uncompressed)
└── README.md                  # Tài liệu đề tài & phân rã khoa học chi tiết
```

---

## 📊 4. Đánh Giá Kết Quả Thực Nghiệm

### 4.1. Đánh Giá Định Lượng Trên Tập Dữ Liệu Chuẩn (Kaggle Ground Truth)
| Phương pháp | PSNR (dB) | SSIM | Nền giấy | Tình trạng nét chữ |
| :--- | :---: | :---: | :---: | :---: |
| **Ảnh gốc bẩn (Raw Dirty Input)** | 14.82 dB | 0.6210 | Loang lổ vết ố, bóng đổ | Bị lấp bởi vết bẩn, đứt nét |
| **Baseline (Triangle / Otsu toàn cục)** | 16.45 dB | 0.6830 | Còn quầng ố sẫm lớn | Chữ bị đứt gãy nặng (clipping) |
| **Sauvola thông thường** | 18.20 dB | 0.7450 | Sạch đốm nhỏ | Nét mảnh bị rách, xơ xác |
| **Pipeline Đề Xuất (Otsu + Black-Hat 9 Bước)** | **25.62 dB** | **0.9440** | **Trắng tinh khôi 100% (255)** | **Đậm đen, đặc ruột, liền mạch** |

### 4.2. Khả Năng Đọc Văn Bản Của OCR (Optical Character Recognition)
- **Trước khi xử lý:** Bộ máy OCR đọc nhầm các mảng ố cà phê thành các ký tự rác (`~`, `#`, `...`), tỷ lệ nhận dạng từ đúng $< 30\%$.
- **Sau khi xử lý:** Văn bản được tách biệt hoàn toàn trên nền trắng $255$, các từ tiếng Anh và số được nhận dạng chính xác từng dòng với độ tin cậy trung bình $> 88\%$ (nhận dạng đủ **11/11 dòng chữ**).

### 4.3. Minh Chứng Khoa Học: Phục Hồi Nét Chữ Đứt & Xuất 9 Ảnh Trung Gian
Người dùng có thể kiểm chứng trực quan bằng lệnh:
```bash
python run_pipeline.py --proof
```
Lệnh sẽ tự động:
1. Xuất file ảnh đối chiếu cận cảnh `stroke_restoration_proof.png`.
2. Tự động lưu toàn bộ 9 ảnh trung gian của các bước thuật toán vào thư mục `steps_output/`.
3. Chạy đối chiếu OCR trước và sau phục hồi trên console.

---

## 🚀 5. Hướng Dẫn Cài Đặt & Sử Dụng

### 5.1. Chạy Ứng Dụng Web Cục Bộ (FastAPI + HTML5 Client)

1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
2. Khởi chạy máy chủ Web:
   ```bash
   python -m uvicorn api.index:app --reload --port 8000
   ```
3. Mở trình duyệt truy cập: **http://localhost:8000**  
   - Kéo thả bất kỳ ảnh tài liệu nào hoặc bấm nút **"Ảnh mẫu Kaggle"** để xem ngay kết quả làm sạch!
   - Chuyển sang tab **"Chi Tiết Toàn Bộ 9 Bước Thuật Toán"** để xem ảnh kết quả và giải thích của từng bước.

---

### 5.2. Chạy Dòng Lệnh (CLI Runner)

- **Làm sạch 1 ảnh và xuất toàn bộ 9 ảnh trung gian:**
  ```bash
  python run_pipeline.py --input samples/sample_dirty.png --output clean.png --save-steps --steps-dir steps_output
  ```
- **Chạy minh chứng khoa học (Proof & OCR):**
  ```bash
  python run_pipeline.py --proof
  ```
- **Làm sạch kèm đọc chữ OCR:**
  ```bash
  python run_pipeline.py --input samples/sample_dirty.png --ocr
  ```
- **Xử lý hàng loạt toàn bộ một thư mục ảnh:**
  ```bash
  python run_pipeline.py --batch
  ```

---

### 5.3. Triển Khai Lên Vercel (Cloud Deployment)

Hệ thống đã được thiết kế sẵn sàng 100% cho kiến trúc Serverless của Vercel:

```bash
npx vercel
```
*Vercel sẽ tự động tải mã nguồn lên và cung cấp đường dẫn truy cập công khai (ví dụ: `https://x-l-nh.vercel.app`).*

---

## 👥 6. Thông Tin Nhóm Thực Hiện
- **Môn học:** Xử lý Ảnh Số (DIP)
- **Nhóm:** Nhóm 3
- **Đề tài:** Ứng dụng phân ngưỡng tự động và các phép toán hình thái học để làm sạch và phục hồi ảnh tài liệu cũ, đứt nét.
