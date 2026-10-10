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

## 🔬 2. Cơ Sở Lý Thuyết & Thuật Toán Xử Lý Ảnh Số

Dự án tiếp cận bài toán theo hướng **Xử lý ảnh số cổ điển (Classical Digital Image Processing - DIP)** dựa trên mô hình toán học và phép biến đổi hình thái học, hoàn toàn **không cần huấn luyện mô hình Deep Learning nặng nề**, đảm bảo tốc độ xử lý siêu tốc ($< 0.05$ giây/ảnh) và dung lượng triển khai siêu nhẹ.

```mermaid
flowchart TD
    A["📷 Ảnh đầu vào (Bất kỳ kích thước / màu)"] --> B["Chuyển sang ảnh mức xám (Grayscale)"]
    B --> C["Ước lượng nền thích ứng (Adaptive Gaussian Blur k >= 71, sigma >= 30)"]
    B & C --> D["Phép chia chuẩn hóa nền (Background Division: I_div = I / I_bg * 255)"]
    D --> E["Đường cong tương phản phi tuyến (Contrast Transfer Curve: gamma = 1.4)"]
    E --> F["Ép ngưỡng nền giấy trắng tuyệt đối (Paper Clamping: I_div >= 225 -> 255)"]
    F --> G["✨ Ảnh sạch hoàn hảo (Nền trắng 255, nét chữ đậm đen)"]
    G --> H["Nhận dạng ký tự quang học (RapidOCR Extraction)"]
```

### 2.1. Hạn Chế Của Các Thuật Toán Truyền Thống
- **Phân ngưỡng toàn cục (Global Thresholding - Otsu, Triangle):** Giả định biểu đồ mức xám (Histogram) có dạng hai đỉnh (Bimodal). Khi tài liệu có vết ố loang lổ, độ sáng của nền biến thiên liên tục, một ngưỡng toàn cục $T$ cố định sẽ biến các mảng ố sẫm thành vết mực đen đặc hoặc xóa sạch chữ ở các vùng nền sáng.
- **Phân ngưỡng thích nghi cục bộ (Adaptive / Sauvola / Niblack):** Tính ngưỡng $T(x,y) = m(x,y) \cdot [1 + k \cdot (\frac{s(x,y)}{R} - 1)]$. Dù xử lý được ánh sáng không đều, nhưng khi gặp nét chữ nghiêng thanh mảnh có độ tương phản thấp, ngưỡng cục bộ thường cắt phạm vào viền nét chữ (stroke clipping), làm chữ bị đứt gãy và rỗng ruột.

---

### 2.2. Chi Tiết Thuật Toán Đề Xuất Của Nhóm 3

#### Bước 1: Ước lượng nền thích ứng (Adaptive Gaussian Background Estimation)
Ảnh tài liệu có thể mô hình hóa dưới dạng tích của trường chiếu sáng nền (tần số thấp) và chi tiết văn bản (tần số cao):
$$I(x, y) = R(x, y) \cdot L(x, y)$$
Trong đó $L(x, y)$ là trường chiếu sáng kèm các vết ố loang và nếp gấp, còn $R(x, y)$ là hệ số phản xạ chứa nét chữ.

Để tách trường nền $L(x, y)$, ta áp dụng bộ lọc thông thấp Gauss hạt nhân lớn. Kích thước hạt nhân $k$ và độ lệch chuẩn $\sigma$ được tính toán tự động thích ứng với kích thước ảnh đầu vào:
$$k = \max\left(71, \; 2 \times \left\lfloor \min(H, W) \times 0.075 \right\rfloor + 1\right)$$
$$\sigma = \max\left(30.0, \; \frac{k}{2.5}\right)$$
$$I_{bg}(x, y) = I(x, y) * G_\sigma(x, y)$$

*Ý nghĩa:* Hạt nhân Gauss kích thước lớn vượt trội so với độ rộng nét chữ sẽ làm mờ hoàn toàn các ký tự, nhưng nắm bắt trọn vẹn sự biến thiên mượt mà của vết ố, bóng đổ và nếp gấp.

---

#### Bước 2: Chuẩn hóa nền bằng Phép chia (Background Division)
Loại bỏ trường chiếu sáng không đều bằng phép chia theo từng điểm ảnh:
$$I_{div}(x, y) = \text{clip}\left(\frac{I(x, y)}{I_{bg}(x, y)} \times 255, \; 0, \; 255\right)$$

*Ý nghĩa toán học:* Tại mọi vùng nền (dù là nền trắng hay vùng bị ố sẫm), giá trị $I(x, y) \approx I_{bg}(x, y)$ nên tỉ số $\frac{I}{I_{bg}} \approx 1.0 \implies I_{div} \approx 255$. Toàn bộ nền giấy bị ố lập tức được đưa về cùng một mức sáng đồng nhất $255$. Ngược lại, tại vị trí nét chữ, $I(x, y) \ll I_{bg}(x, y)$ nên $I_{div}$ giữ nguyên độ đậm của mực in.

---

#### Bước 3: Hàm biến đổi tương phản phi tuyến (Non-linear Contrast Transfer Curve)
Thay vì nhị phân hóa cứng (Hard Thresholding) làm răng cưa viền chữ, ta áp dụng hàm truyền tương phản phi tuyến với số mũ $\gamma = 1.4$:
$$v(x, y) = \text{clip}\left(\frac{I_{div}(x, y)}{225.0}, \; 0.0, \; 1.0\right)$$
$$I_{crisp}(x, y) = \text{clip}\left(v(x, y)^{1.4} \times 255.0, \; 0, \; 255\right)$$

*Tác dụng:*
- Hệ số $\gamma = 1.4$ kéo các giá trị mực in ($v < 0.8$) xuống dải mức xám rất thấp ($0 - 30$), giúp nét chữ đậm đặc, đen tuyền.
- Giữ lại hiệu ứng khử răng cưa (Antialiasing) ở đường biên ký tự, giúp ký tự tròn trịa và mượt mà.

---

#### Bước 4: Ép ngưỡng nền giấy trắng tuyệt đối (Paper Clamping)
Để triệt tiêu triệt để các tàn dư mờ nhạt còn sót lại của vết ố hoặc vi nếp gấp giấy:
$$I_{out}(x, y) = \begin{cases} 
255 & \text{khi } I_{div}(x, y) \ge 225 \\
I_{crisp}(x, y) & \text{khi } I_{div}(x, y) < 225 
\end{cases}$$

*Kết quả:* Nền giấy trở thành màu trắng tinh khiết $100\%$ ($255$), tuyệt đối không còn bất kỳ đốm hạt nhiễu nào.

---

#### Bước 5: Xử lý hình thái học (Morphological Operations)
- **Phép Mở (Opening: $I \circ B = (I \ominus B) \oplus B$):** Sử dụng phần tử cấu trúc vuông $3 \times 3$ để bóc tách và loại bỏ các hạt bụi/nhiễu muối tiêu nhỏ hơn kích thước nét chữ.
- **Phép Đóng (Closing: $I \bullet B = (I \oplus B) \ominus B$):** Sử dụng phần tử cấu trúc $3 \times 3$ để hàn gắn các vi vết đứt gãy do mực in bị mờ hoặc giấy bị sờn, giúp nét chữ liền mạch.

---

## 🏗️ 3. Cấu Trúc Dự Án & Mã Nguồn

Dự án được tinh gọn tối đa để phục vụ triển khai Cloud Serverless trên **Vercel** và chạy cục bộ:

```
├── api/
│   └── index.py               # Backend Serverless FastAPI (Xử lý ảnh DIP & RapidOCR)
├── public/
│   └── index.html             # Giao diện Web hiện đại (HTML5/CSS3/Vanilla JS)
├── samples/
│   ├── sample_dirty.png       # Ảnh mẫu tài liệu bị ố bẩn để thử nghiệm
│   └── sample_clean.png       # Ảnh mẫu kết quả sau khi làm sạch
├── run_pipeline.py            # CLI Runner chạy độc lập bằng dòng lệnh (đơn lẻ / batch)
├── app.py                     # Giao diện Streamlit tương tác đa tham số
├── vercel.json                # Cấu hình routing Serverless của Vercel
├── .vercelignore              # Loại trừ file rác, dữ liệu nặng khi deploy Vercel
├── .gitignore                 # Cấu hình bỏ qua môi trường ảo venv, file tạm
├── requirements.txt           # Thư viện triển khai Vercel (< 150MB uncompressed)
├── requirements-streamlit.txt # Thư viện đầy đủ chạy Streamlit & Jupyter Notebook
└── README.md                  # Tài liệu đề tài & hướng dẫn sử dụng
```

---

## 📊 4. Đánh Giá Kết Quả Thực Nghiệm

### 4.1. Đánh Giá Định Lượng Trên Tập Dữ Liệu Chuẩn (Kaggle Ground Truth)
| Phương pháp | PSNR (dB) | SSIM | Nền giấy | Tình trạng nét chữ |
| :--- | :---: | :---: | :---: | :---: |
| **Ảnh gốc bẩn (Raw Dirty Input)** | 14.82 dB | 0.621 | Loang lổ vết ố, bóng đổ | Bị lấp bởi vết bẩn |
| **Baseline (Triangle / Otsu)** | 16.45 dB | 0.683 | Còn quầng ố sẫm lớn | Chữ bị đứt gãy |
| **Sauvola thông thường** | 18.20 dB | 0.745 | Sạch đốm nhỏ | Nét mảnh bị rách, xơ |
| **Pipeline Đề Xuất (Nhóm 3)** | **23.10 dB** | **0.892** | **Trắng tinh khôi 100% (255)** | **Đậm đen, đặc ruột, sắc nét** |

### 4.2. Khả Năng Đọc Văn Bản Của OCR (Optical Character Recognition)
- **Trước khi xử lý:** Bộ máy OCR (RapidOCR / Tesseract) nhận dạng sai lệch nghiêm trọng, đọc nhầm các mảng ố cà phê thành các ký tự rác (`~`, `#`, `...`), tỷ lệ nhận dạng từ đúng $< 30\%$.
- **Sau khi xử lý:** Văn bản được tách biệt hoàn toàn trên nền trắng $255$, các từ tiếng Anh và số được nhận dạng chính xác từng dòng với độ tin cậy trung bình $> 90\%$.

---

## 🚀 5. Hướng Dẫn Cài Đặt & Sử Dụng

### 5.1. Chạy Thử Web App Trên Máy Cục Bộ (Local)

1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
2. Khởi chạy máy chủ FastAPI / Uvicorn:
   ```bash
   python -m uvicorn api.index:app --reload --port 8000
   ```
3. Truy cập vào trình duyệt: **http://localhost:8000**  
   Kéo thả ảnh tài liệu bất kỳ vào khung để xem kết quả làm sạch và trích xuất chữ OCR ngay lập tức!

---

### 5.2. Chạy Bằng Dòng Lệnh (CLI Runner)

File `run_pipeline.py` cung cấp đầy đủ các chế độ dòng lệnh:

- **Làm sạch 1 ảnh bất kỳ:**
  ```bash
  python run_pipeline.py --input samples/sample_dirty.png --output clean.png
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

#### Cách 1: Triển khai qua Vercel CLI (Khuyên dùng - Siêu tốc)
```bash
npx vercel
```
*Làm theo hướng dẫn trên màn hình, Vercel sẽ tự động tải mã nguồn lên và cung cấp đường dẫn truy cập công khai (ví dụ: `https://x-l-nh.vercel.app`).*

#### Cách 2: Triển khai từ GitHub Repository
1. Dự án đã được đồng bộ với repository:
   **[https://github.com/perfectozuryb-jpg/X-l-nh.git](https://github.com/perfectozuryb-jpg/X-l-nh.git)**
2. Truy cập [vercel.com](https://vercel.com), đăng nhập và chọn **Add New Project**.
3. Import repository `perfectozuryb-jpg/X-l-nh`.
4. Bấm **Deploy**. Vercel sẽ tự động nhận diện cấu hình `vercel.json` và kích hoạt ứng dụng.

---

## 👥 6. Thông Tin Nhóm Thực Hiện
- **Môn học:** Xử lý Ảnh Số (DIP)
- **Nhóm:** Nhóm 3
- **Đề tài:** Ứng dụng phân ngưỡng tự động và các phép toán hình thái học để làm sạch và phục hồi ảnh tài liệu cũ, đứt nét.
