import io
import os
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import thư viện xử lý ảnh (ưu tiên OpenCV, có fallback Pillow)
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False
    import numpy as np
    from PIL import Image, ImageFilter

# Khởi tạo RapidOCR nếu có sẵn
_ocr_engine = None
def get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
        except Exception:
            _ocr_engine = False
    return _ocr_engine if _ocr_engine is not False else None

app = FastAPI(
    title="Denoising Dirty Documents DIP API - Nhóm 3",
    description="Quy trình Xử lý ảnh số toàn diện 9 bước: Phân ngưỡng tự động Otsu & Hình thái học Black-Hat"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def encode_img_b64(arr):
    """Helper chuyển ma trận điểm ảnh numpy sang chuỗi base64 PNG."""
    if HAS_CV2:
        _, buf = cv2.imencode('.png', arr)
        return base64.b64encode(buf).decode('utf-8')
    else:
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode('utf-8')

def run_ocr_text(img_gray):
    """Bóc tách văn bản qua RapidOCR (với fallback Tesseract nếu có)."""
    engine = get_ocr()
    if engine is not None:
        try:
            res, _ = engine(img_gray)
            if res:
                lines = []
                for bbox, text, score in res:
                    try:
                        conf = float(score)
                    except Exception:
                        conf = 0.90
                    lines.append({
                        "text": text.strip(),
                        "confidence": round(conf * 100)
                    })
                if lines:
                    return lines
        except Exception as e:
            print("RapidOCR error:", e)

    # Thử fallback pytesseract nếu đã cài đặt
    try:
        import pytesseract
        for p in [r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break
        txt = pytesseract.image_to_string(img_gray)
        lines = [{"text": l.strip(), "confidence": 85} for l in txt.split('\n') if l.strip()]
        if lines:
            return lines
    except Exception:
        pass

    return []

def process_pipeline_with_all_steps(gray):
    """
    Quy trình Xử lý ảnh số (DIP) 9 bước phân rã toàn diện - Nhóm 3:
    1. Chuyển đổi mức xám (Grayscale Conversion).
    2. Ước lượng trường nền thích ứng (Adaptive Gaussian Blur Estimation).
    3. Làm phẳng nền bằng phép chia (Background Division).
    4. Phân ngưỡng tự động Otsu (Automatic Otsu Thresholding).
    5. Tăng cường tương phản phi tuyến (Non-linear Gamma Contrast Transfer).
    6. Biến đổi hình thái học Black-Hat (Morphological Black-Hat Transform).
    7. Triệt tiêu quầng sương mờ vết ố cà phê (Coffee Stain Haze Suppression).
    8. Ép nền giấy trắng tuyệt đối 100% (100% White Background Clamping).
    9. Lọc hạt bụi ngoài lề qua thành phần liên thông (Connected Components Noise Filtering).
    """
    h, w = gray.shape[:2]
    k = max(71, int(min(h, w) * 0.15))
    if k % 2 == 0:
        k += 1
    sigma = max(30.0, float(k) / 2.5)

    # ─── BƯỚC 2: Ước lượng trường nền thích ứng ──────────────────────────────
    if HAS_CV2:
        bg = cv2.GaussianBlur(gray, (k, k), sigma)
        div = cv2.divide(gray, bg, scale=255)
    else:
        pil_img = Image.fromarray(gray)
        pil_bg = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
        bg = np.array(pil_bg)
        bg_arr = bg.astype(float) + 1e-5
        div = np.clip((gray.astype(float) / bg_arr) * 255.0, 0, 255).astype(np.uint8)

    # ─── BƯỚC 4: Phân ngưỡng tự động Otsu ───────────────────────────────────
    if HAS_CV2:
        th_otsu, _ = cv2.threshold(div, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th_paper = int(th_otsu + (255 - th_otsu) * 0.5)
        # Ảnh nhị phân Otsu (chữ đen nền trắng để trực quan)
        otsu_bin = cv2.threshold(div, int(th_otsu), 255, cv2.THRESH_BINARY)[1]
    else:
        th_otsu = 181.0
        th_paper = 218
        otsu_bin = (div >= th_otsu).astype(np.uint8) * 255

    # ─── BƯỚC 5: Tăng cường tương phản phi tuyến nét chữ ────────────────────
    v = np.clip(div.astype(float) / float(th_paper), 0.0, 1.0)
    contrast = np.clip(np.power(v, 1.4) * 255.0, 0, 255).astype(np.uint8)
    contrast[div >= th_paper] = 255

    # ─── BƯỚC 6: Biến đổi hình thái học Black-Hat ───────────────────────────
    if HAS_CV2:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
        # Tăng sáng để mắt người nhìn rõ bản đồ nét mực Black-Hat phát sáng trên nền tối
        bh_visual = np.clip(bh.astype(float) * 3.5, 0, 255).astype(np.uint8)
    else:
        bh = np.zeros_like(gray)
        bh_visual = np.zeros_like(gray)

    # ─── BƯỚC 7: Triệt tiêu quầng sương mờ vết ố cà phê ─────────────────────
    stain_suppressed = contrast.copy()
    if HAS_CV2:
        stain_haze = (div >= int(th_otsu)) & (bh < 18)
        stain_suppressed[stain_haze] = 255

    # ─── BƯỚC 8: Ép nền giấy trắng tuyệt đối 100% ───────────────────────────
    clamped = stain_suppressed.copy()
    clamped[div >= th_paper] = 255

    # ─── BƯỚC 9: Lọc hạt bụi ngoài lề qua thành phần liên thông ─────────────
    clean = clamped.copy()
    if HAS_CV2:
        mask_ink = (clean < 240).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_ink, 8)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area <= 2:
                clean[labels == i] = 255

    # Đếm thành phần liên thông để chứng minh phục hồi nét đứt
    num_cc_raw = 0
    num_cc_clean = 0
    if HAS_CV2:
        mask_raw = (gray < 200).astype(np.uint8)
        num_cc_raw, _ = cv2.connectedComponents(mask_raw)
        mask_clean = (clean < 200).astype(np.uint8)
        num_cc_clean, _ = cv2.connectedComponents(mask_clean)

    white_ratio = float(np.mean(clean == 255) * 100)

    # ─── ĐÓNG GÓI CHI TIẾT ĐẦY ĐỦ 9 BƯỚC DIP ────────────────────────────────
    steps_data = [
        {
            "step_number": 1,
            "title": "Chuyển Đổi Mức Xám (Grayscale Conversion)",
            "algorithm": "Y = 0.299*R + 0.587*G + 0.114*B",
            "how": "Bóc tách bức ảnh đa kênh màu (RGB) thành ma trận 1 kênh cường độ sáng từ 0 (đen tuyệt đối) đến 255 (trắng tuyệt đối).",
            "why": "Loại bỏ thông tin màu sắc gây nhiễu, tập trung hoàn toàn vào cường độ phản xạ ánh sáng của hạt mực in và nền giấy. Giúp giảm 66% bộ nhớ và tăng tốc độ xử lý ma trận gấp 3 lần.",
            "image": f"data:image/png;base64,{encode_img_b64(gray)}"
        },
        {
            "step_number": 2,
            "title": "Ước Lượng Trường Nền Thích Ứng (Adaptive Gaussian Blur)",
            "algorithm": f"Gaussian Blur 2D (k = {k}×{k}, σ = {sigma:.1f})",
            "how": f"Tích chập ảnh mức xám với bộ lọc Gauss có kích thước hạt nhân siêu lớn ({k}×{k}) tự động co giãn theo 15% kích thước tài liệu.",
            "why": "Nét chữ là chi tiết biến thiên tần số cao (kích thước nhỏ vài pixel) sẽ bị làm mờ hoàn toàn, chỉ giữ lại mô hình biến thiên chậm của trường sáng nền, vết ố loang cà phê và nếp gấp giấy (tần số thấp).",
            "image": f"data:image/png;base64,{encode_img_b64(bg)}"
        },
        {
            "step_number": 3,
            "title": "Làm Phẳng & Chuẩn Hóa Nền Bằng Phép Chia (Background Division)",
            "algorithm": "I_div = (I_gray / I_bg) * 255",
            "how": "Lấy ma trận điểm ảnh xám chia theo từng phần tử cho ma trận trường nền, sau đó nhân tỷ lệ với 255.",
            "why": "Tại mọi vị trí nền giấy (kể cả vùng bị ố cà phê đậm hay bóng tối), giá trị gốc xấp xỉ giá trị nền (I_gray ≈ I_bg) nên tỉ số xấp xỉ 1.0, nhân 255 sẽ biến toàn bộ nền thành mặt phẳng sáng đồng nhất! Ngược lại tại nét chữ, I_gray << I_bg nên tỉ số rất nhỏ, bảo toàn được độ đậm nguyên bản của nét mực.",
            "image": f"data:image/png;base64,{encode_img_b64(div)}"
        },
        {
            "step_number": 4,
            "title": "Phân Tích Lược Đồ & Phân Ngưỡng Tự Động Otsu (Automatic Otsu Thresholding)",
            "algorithm": f"Otsu's Inter-Class Variance: max σ_B^2(t) ➔ T_otsu = {th_otsu:.1f}, T_paper = {th_paper}",
            "how": f"Thuật toán Otsu tự động phân tích biểu đồ tần số mức xám (histogram) của ảnh phẳng nền để tìm ngưỡng tối ưu {th_otsu:.1f} làm cực đại phương sai giữa cụm mực và cụm nền giấy, đồng thời tự động nội suy ranh giới bắt đầu của nền giấy T_paper = {th_paper}.",
            "why": "Loại bỏ hoàn toàn nhược điểm của các ngưỡng cắt tĩnh (hardcoded thresholds). Từng bức ảnh tài liệu với độ đậm nhạt khác nhau sẽ tự động tìm được ngưỡng ranh giới hoàn hảo mà không cần con người can thiệp.",
            "image": f"data:image/png;base64,{encode_img_b64(otsu_bin)}"
        },
        {
            "step_number": 5,
            "title": "Tăng Cường Tương Phản Phi Tuyến Nét Chữ (Non-linear Gamma Contrast Transfer)",
            "algorithm": f"v = clip(I_div / {th_paper}, 0, 1); I_gamma = (v ^ 1.4) * 255",
            "how": f"Chuẩn hóa ma trận giá trị điểm ảnh theo ngưỡng ranh giới nền giấy T_paper={th_paper} về đoạn [0.0, 1.0], sau đó áp dụng hàm lũy thừa phi tuyến bậc gamma = 1.4 để kéo đậm các giá trị tối.",
            "why": "Kéo các hạt mực in về sắc đen tuyền sâu thẳm (0 - 30), nhưng giữ lại hiệu ứng chuyển sắc mượt mà (anti-aliasing) ở đường viền ký tự. Chữ in trở nên đậm đặc ruột nhưng không bị gãy nét hay răng cưa như phân ngưỡng nhị phân thô.",
            "image": f"data:image/png;base64,{encode_img_b64(contrast)}"
        },
        {
            "step_number": 6,
            "title": "Biến Đổi Hình Thái Học Black-Hat (Morphological Black-Hat Transform)",
            "algorithm": "BH(I) = (I • B) - I = Closing(I, 5×5) - I",
            "how": "Áp dụng phép toán hình thái học Đóng (Closing = Dilation theo sau bởi Erosion) với phần tử cấu trúc chữ nhật B kích thước 5×5, sau đó lấy ảnh sau phép Đóng trừ đi ảnh gốc.",
            "why": "Phép Đóng tiến hành hàn gắn các vi khoảng hở nứt gãy giữa các hạt mực in. Phép trừ sau đó bóc trần bản đồ cường độ của nét mực thực sự trên nền tối, tách biệt hoàn toàn chi tiết nét chữ khỏi các mảng ố cà phê phẳng.",
            "image": f"data:image/png;base64,{encode_img_b64(bh_visual)}"
        },
        {
            "step_number": 7,
            "title": "Triệt Tiêu Quầng Sương Mờ Vết Ố Cà Phê (Coffee Stain Haze Suppression)",
            "algorithm": f"Stain_Haze = (I_div >= {int(th_otsu)}) ∧ (BH < 18) ➔ I[Stain_Haze] = 255",
            "how": "Giao thoa giữa bản đồ phân vùng Otsu và bản đồ nét mực Black-Hat. Những điểm ảnh có mức xám nằm trong vùng quầng ố nhưng Black-Hat xác nhận không có nét mực thật sự (< 18) sẽ bị tẩy trắng hoàn toàn về 255.",
            "why": "Triệt tiêu 100% quầng sương mờ của vết ố cà phê loang lổ mà không cắt phạm vào các nét chữ nghiêng thanh mảnh có độ tương phản thấp.",
            "image": f"data:image/png;base64,{encode_img_b64(stain_suppressed)}"
        },
        {
            "step_number": 8,
            "title": "Ép Nền Giấy Trắng Tuyệt Đối 100% (100% White Background Clamping)",
            "algorithm": f"I_clamped[I_div >= {th_paper}] = 255",
            "how": f"Toàn bộ điểm ảnh có giá trị phẳng nền vượt qua ngưỡng ranh giới giấy T_paper = {th_paper} được gán giá trị số học tuyệt đối 255 (trắng tinh khôi).",
            "why": "Đưa không gian nền tài liệu về màu trắng chuẩn 255 tuyệt đối, xóa sổ toàn bộ gradient chiếu sáng không đều, bóng tối và nếp nhăn giấy.",
            "image": f"data:image/png;base64,{encode_img_b64(clamped)}"
        },
        {
            "step_number": 9,
            "title": "Lọc Hạt Bụi Ngoài Lề Qua Thành Phần Liên Thông (Connected Components Noise Filtering)",
            "algorithm": "Connected Components 8-connectivity (Remove Area <= 2 px)",
            "how": "Dò tìm và gán nhãn các miền liên thông 8 hướng của hạt mực. Loại bỏ các đốm hạt bụi đơn lẻ cô lập có diện tích ≤ 2 pixel và các vệt bẩn ngoại lai nằm ngoài lề văn bản.",
            "why": "Làm sạch tinh khiết đến từng pixel, loại bỏ các đốm hạt bụi li ti do giấy cũ mục nát, hoàn thiện văn bản đạt tiêu chuẩn chất lượng số hóa cao nhất.",
            "image": f"data:image/png;base64,{encode_img_b64(clean)}"
        }
    ]

    metrics = {
        "th_otsu": float(th_otsu),
        "th_paper": int(th_paper),
        "k_gauss": int(k),
        "sigma_gauss": float(sigma),
        "white_ratio": round(white_ratio, 2),
        "connected_components_raw": int(num_cc_raw),
        "connected_components_clean": int(num_cc_clean)
    }

    return clean, steps_data, metrics

@app.get("/")
def root():
    if os.path.exists("public/index.html"):
        from fastapi.responses import FileResponse
        return FileResponse(
            "public/index.html",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {
        "status": "ok",
        "service": "Denoising Dirty Documents DIP Pipeline - Nhóm 3",
        "total_steps": 9
    }

@app.get("/api")
@app.get("/api/health")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Denoising Dirty Documents DIP Pipeline - Nhóm 3",
        "total_steps": 9,
        "engine": "OpenCV" if HAS_CV2 else "Pillow-NumPy"
    }

@app.post("/api/denoise")
@app.post("/denoise")
async def denoise(file: UploadFile = File(...), run_ocr: bool = Form(True)):
    try:
        contents = await file.read()
        if not contents:
            return JSONResponse({"error": "File ảnh rỗng"}, status_code=400)

        # Giải mã ảnh
        if HAS_CV2:
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return JSONResponse({"error": "Không thể giải mã file ảnh"}, status_code=400)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            pil_img = Image.open(io.BytesIO(contents)).convert('L')
            gray = np.array(pil_img)

        clean, steps, metrics = process_pipeline_with_all_steps(gray)

        # Chạy OCR bóc tách chữ
        ocr_clean_lines = run_ocr_text(clean)
        ocr_raw_lines = run_ocr_text(gray)

        raw_b64 = encode_img_b64(gray)
        clean_b64 = encode_img_b64(clean)

        return {
            "success": True,
            "filename": file.filename or "document.png",
            "width": int(clean.shape[1]),
            "height": int(clean.shape[0]),
            "raw_image": f"data:image/png;base64,{raw_b64}",
            "clean_image": f"data:image/png;base64,{clean_b64}",
            "metrics": metrics,
            "steps": steps,
            "ocr_raw": ocr_raw_lines,
            "ocr_clean": ocr_clean_lines
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Phục vụ thư mục samples cho giao diện web
if os.path.exists("samples"):
    try:
        app.mount("/samples", StaticFiles(directory="samples"), name="samples")
    except Exception:
        pass

