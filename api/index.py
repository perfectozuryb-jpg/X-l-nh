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
    description="Quy trình Xử lý ảnh số toàn diện 9 bước: Bản 1 (Gaussian Blur) & Bản 2 (Median Filter)"
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

def encode_step_preview_b64(arr, max_width=800, quality=82):
    """Nén ảnh xem trước cho 9 bước trung gian (JPEG) để tránh vượt giới hạn 4.5MB của Vercel."""
    h, w = arr.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        new_size = (max_width, max(1, int(h * scale)))
        if HAS_CV2:
            arr = cv2.resize(arr, new_size, interpolation=cv2.INTER_AREA)
        else:
            arr = np.array(Image.fromarray(arr).resize(new_size))

    if HAS_CV2:
        _, buf = cv2.imencode('.jpg', arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
    else:
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

def run_ocr_text(img_gray):
    """Bóc tách văn bản tiếng Anh qua RapidOCR (fallback pytesseract nếu có)."""
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
                    conf_pct = round(conf * 100)
                    txt = text.strip()
                    if txt and conf_pct >= 40:
                        lines.append({
                            "text": txt,
                            "confidence": conf_pct
                        })
                if lines:
                    return lines
        except Exception as e:
            print("RapidOCR error:", e)

    try:
        import pytesseract
        for p in [r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break
        txt = pytesseract.image_to_string(img_gray, lang="eng")
        lines = [{"text": l.strip(), "confidence": 85} for l in txt.split('\n') if len(l.strip()) >= 2]
        if lines:
            return lines
    except Exception:
        pass

    return []

def filter_dust_and_isolated_specks(clean):
    """
    Bước 9 nâng cao: Lọc hạt bụi nhỏ, vết xám mờ quanh chữ và các chấm bẩn rơi lẻ giữa các dòng
    mà vẫn bảo toàn 100% nét chữ mảnh và dấu chấm câu / dấu chấm chữ i.
    """
    if not HAS_CV2:
        return clean

    h, w = clean.shape[:2]
    out = clean.copy()
    mask_ink = (out < 225).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_ink, 8)
    dark_mask = (out < 175).astype(np.uint8)

    tiny_area = max(3, int((h * w) / 60000))
    floating_dot_max = max(20, int((h * w) / 7000))
    speckle_max = max(40, int((h * w) / 3500))
    hx = max(10, int(w * 0.022))

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh_b = stats[i, cv2.CC_STAT_HEIGHT]
        roi_mask = (labels[y:y+bh_b, x:x+bw] == i)
        roi_vals = out[y:y+bh_b, x:x+bw][roi_mask]
        mean_val = float(np.mean(roi_vals))
        min_val = float(np.min(roi_vals))

        # 1) Hạt bụi siêu nhỏ <= 2 px
        if area <= 2:
            out[y:y+bh_b, x:x+bw][roi_mask] = 255
            continue

        # 2) Hạt bụi xám nhỏ (không có lõi mực tối)
        if area <= tiny_area and min_val > 110:
            out[y:y+bh_b, x:x+bw][roi_mask] = 255
            continue

        # 3) Mảng vết ố xám rất nhạt sót lại quanh chữ (CHỈ xóa khi hoàn toàn không có lõi mực đậm: min_val > 145)
        if area <= speckle_max and min_val > 145 and mean_val > 180:
            out[y:y+bh_b, x:x+bw][roi_mask] = 255
            continue

        # 4) Chấm bẩn cô lập nằm lơ lửng trong khoảng trắng giữa các dòng
        #    (không có chữ lân cận theo phương ngang và không nằm trên/dưới thân chữ như dấu chấm chữ i)
        if area <= floating_dot_max and max(bw, bh_b) <= max(10, int(min(h, w) * 0.04)):
            x0, x1 = max(0, x - hx), min(w, x + bw + hx)
            y0, y1 = max(0, y - 1), min(h, y + bh_b + 1)
            band_dark = int(np.sum(dark_mask[y0:y1, x0:x1])) - int(np.sum(dark_mask[y:y+bh_b, x:x+bw][roi_mask]))

            v_gap = max(4, int(min(h, w) * 0.016))
            bx0, bx1 = max(0, x - 2), min(w, x + bw + 2)
            below_dark = int(np.sum(dark_mask[min(h, y + bh_b):min(h, y + bh_b + v_gap), bx0:bx1]))
            above_dark = int(np.sum(dark_mask[max(0, y - v_gap):y, bx0:bx1]))

            if band_dark <= 2 and below_dark == 0 and above_dark == 0:
                out[y:y+bh_b, x:x+bw][roi_mask] = 255
                continue

        # 5) Mảng bóng xám rất nhạt bất kỳ kích thước nào (không có điểm mực đậm)
        if mean_val > 195 and min_val > 155:
            out[y:y+bh_b, x:x+bw][roi_mask] = 255

    return out

def process_pipeline_gaussian(gray):
    """Bản 1: Quy trình 9 bước gốc dùng Adaptive Gaussian Blur (đồng bộ với Word & PPT)."""
    h, w = gray.shape[:2]
    k = max(71, int(min(h, w) * 0.15))
    if k % 2 == 0:
        k += 1
    sigma = max(30.0, float(k) / 2.5)

    # Bước 2 & 3: Adaptive Gaussian Blur + Background Division
    if HAS_CV2:
        bg = cv2.GaussianBlur(gray, (k, k), sigma)
        div = cv2.divide(gray, np.maximum(bg, 1), scale=255)
    else:
        pil_img = Image.fromarray(gray)
        pil_bg = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
        bg = np.array(pil_bg)
        bg_arr = bg.astype(float) + 1e-5
        div = np.clip((gray.astype(float) / bg_arr) * 255.0, 0, 255).astype(np.uint8)

    # Bước 4: Phân ngưỡng tự động Otsu
    if HAS_CV2:
        th_otsu, _ = cv2.threshold(div, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th_paper = int(th_otsu + (255 - th_otsu) * 0.48)
        otsu_bin = cv2.threshold(div, int(th_otsu), 255, cv2.THRESH_BINARY)[1]
    else:
        th_otsu = 181.0
        th_paper = 216
        otsu_bin = (div >= th_otsu).astype(np.uint8) * 255

    # Bước 5: Tăng cường tương phản phi tuyến Gamma = 1.6 (chữ đậm rõ hơn cho OCR)
    v = np.clip(div.astype(np.float32) / float(max(th_paper, 1)), 0.0, 1.0)
    contrast = np.clip(np.power(v, 1.6) * 255.0, 0, 255).astype(np.uint8)
    contrast[div >= th_paper] = 255

    # Bước 6: Biến đổi hình thái học Black-Hat 5x5
    if HAS_CV2:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bh = cv2.morphologyEx(div, cv2.MORPH_BLACKHAT, se)
        bh_visual = np.clip(bh.astype(np.float32) * 3.5, 0, 255).astype(np.uint8)
    else:
        bh = np.zeros_like(gray)
        bh_visual = np.zeros_like(gray)

    # Bước 7: Triệt tiêu quầng sương mờ vết ố cà phê & viền vòng ố
    stain_suppressed = contrast.copy()
    if HAS_CV2:
        stain_haze = ((div >= int(th_otsu)) & (bh < 18)) | ((div >= int(th_otsu - 25)) & (bh < 10))
        stain_suppressed[stain_haze] = 255

    # Bước 8: Ép nền giấy trắng tuyệt đối 100%
    clamped = stain_suppressed.copy()
    clamped[div >= th_paper] = 255

    # Bước 9: Lọc hạt bụi & chấm bẩn sót lại quanh chữ qua Thành phần liên thông
    clean = filter_dust_and_isolated_specks(clamped)

    num_cc_raw, num_cc_clean = 0, 0
    if HAS_CV2:
        num_cc_raw, _ = cv2.connectedComponents((gray < 200).astype(np.uint8))
        num_cc_clean, _ = cv2.connectedComponents((clean < 200).astype(np.uint8))

    white_ratio = float(np.mean(clean == 255) * 100)

    steps_data = [
        {
            "step_number": 1,
            "title": "Chuyển Đổi Mức Xám (Grayscale Conversion)",
            "algorithm": "Y = 0.299*R + 0.587*G + 0.114*B",
            "how": "Bóc tách bức ảnh đa kênh màu (RGB) thành ma trận 1 kênh cường độ sáng từ 0 (đen tuyệt đối) đến 255 (trắng tuyệt đối).",
            "why": "Loại bỏ thông tin màu sắc gây nhiễu, tập trung hoàn toàn vào cường độ phản xạ ánh sáng của hạt mực in và nền giấy. Giúp giảm 66% bộ nhớ và tăng tốc độ xử lý ma trận gấp 3 lần.",
            "image": encode_step_preview_b64(gray)
        },
        {
            "step_number": 2,
            "title": "Ước Lượng Trường Nền Thích Ứng (Adaptive Gaussian Blur)",
            "algorithm": f"Gaussian Blur 2D (k = {k}×{k}, σ = {sigma:.1f})",
            "how": f"Tích chập ảnh mức xám với bộ lọc Gauss có kích thước hạt nhân siêu lớn ({k}×{k}) tự động co giãn theo 15% kích thước tài liệu.",
            "why": "Nét chữ là chi tiết biến thiên tần số cao (kích thước nhỏ vài pixel) sẽ bị làm mờ hoàn toàn, chỉ giữ lại mô hình biến thiên chậm của trường sáng nền, vết ố loang cà phê và nếp gấp giấy (tần số thấp).",
            "image": encode_step_preview_b64(bg)
        },
        {
            "step_number": 3,
            "title": "Làm Phẳng & Chuẩn Hóa Nền Bằng Phép Chia (Background Division)",
            "algorithm": "I_div = (I_gray / I_bg) * 255",
            "how": "Lấy ma trận điểm ảnh xám chia theo từng phần tử cho ma trận trường nền, sau đó nhân tỷ lệ với 255.",
            "why": "Tại mọi vị trí nền giấy (kể cả vùng bị ố cà phê đậm hay bóng tối), giá trị gốc xấp xỉ giá trị nền (I_gray ≈ I_bg) nên tỉ số xấp xỉ 1.0, nhân 255 sẽ biến toàn bộ nền thành mặt phẳng sáng đồng nhất; còn nét chữ I_gray << I_bg vẫn giữ nguyên độ đậm.",
            "image": encode_step_preview_b64(div)
        },
        {
            "step_number": 4,
            "title": "Phân Tích Lược Đồ & Phân Ngưỡng Tự Động Otsu (Automatic Otsu Thresholding)",
            "algorithm": f"Otsu Inter-Class Variance: max σ_B^2(t) -> T_otsu = {th_otsu:.1f}, T_paper = {th_paper}",
            "how": f"Thuật toán Otsu tự động phân tích histogram của ảnh phẳng nền để tìm ngưỡng tối ưu {th_otsu:.1f} làm cực đại phương sai giữa cụm mực và cụm nền giấy, đồng thời nội suy ranh giới giấy T_paper = {th_paper}.",
            "why": "Loại bỏ hoàn toàn nhược điểm của các ngưỡng cắt tĩnh. Từng bức ảnh tài liệu với độ đậm nhạt khác nhau tự động tìm được ngưỡng ranh giới phù hợp.",
            "image": encode_step_preview_b64(otsu_bin)
        },
        {
            "step_number": 5,
            "title": "Tăng Cường Tương Phản Phi Tuyến Nét Chữ (Non-linear Gamma Contrast Transfer)",
            "algorithm": f"v = clip(I_div / {th_paper}, 0, 1); I_gamma = (v ^ 1.6) * 255",
            "how": f"Chuẩn hóa ma trận giá trị điểm ảnh theo ngưỡng ranh giới nền giấy T_paper={th_paper} về đoạn [0.0, 1.0], sau đó áp dụng hàm lũy thừa phi tuyến bậc gamma = 1.6 để kéo đậm các giá trị tối.",
            "why": "Kéo các hạt mực in về sắc đen sâu nhưng giữ lại chuyển sắc mượt (anti-aliasing) ở đường viền ký tự, giúp chữ đậm liền mạch mà không bị răng cưa như nhị phân thô.",
            "image": encode_step_preview_b64(contrast)
        },
        {
            "step_number": 6,
            "title": "Biến Đổi Hình Thái Học Black-Hat (Morphological Black-Hat Transform)",
            "algorithm": "BH(I) = (I • B) - I = Closing(I, 5×5) - I",
            "how": "Áp dụng phép toán hình thái học Đóng (Closing = Dilation theo sau bởi Erosion) với phần tử cấu trúc chữ nhật B kích thước 5×5, sau đó lấy ảnh sau phép Đóng trừ đi ảnh gốc.",
            "why": "Phép Đóng lấp đầy các nét chữ nhỏ; phép trừ sau đó bóc trần bản đồ cường độ của nét mực thực sự trên nền tối, tách biệt hoàn toàn chi tiết nét chữ khỏi các mảng ố cà phê phẳng.",
            "image": encode_step_preview_b64(bh_visual)
        },
        {
            "step_number": 7,
            "title": "Triệt Tiêu Quầng Sương Mờ Vết Ố Cà Phê (Coffee Stain Haze Suppression)",
            "algorithm": f"Stain_Haze = (I_div >= {int(th_otsu)}) AND (BH < 18) -> I[Stain_Haze] = 255",
            "how": "Giao thoa giữa bản đồ phân vùng Otsu và bản đồ nét mực Black-Hat. Những điểm ảnh có mức xám nằm trong vùng quầng ố nhưng Black-Hat xác nhận không có nét mực thật sự (< 18) sẽ bị tẩy trắng hoàn toàn về 255.",
            "why": "Triệt tiêu quầng sương mờ của vết ố cà phê loang lổ mà không cắt phạm vào các nét chữ nghiêng thanh mảnh có độ tương phản thấp.",
            "image": encode_step_preview_b64(stain_suppressed)
        },
        {
            "step_number": 8,
            "title": "Ép Nền Giấy Trắng Tuyệt Đối 100% (100% White Background Clamping)",
            "algorithm": f"I_clamped[I_div >= {th_paper}] = 255",
            "how": f"Toàn bộ điểm ảnh có giá trị phẳng nền vượt qua ngưỡng ranh giới giấy T_paper = {th_paper} được gán giá trị số học tuyệt đối 255 (trắng tinh khôi).",
            "why": "Đưa không gian nền tài liệu về màu trắng chuẩn 255 tuyệt đối, xóa sổ toàn bộ gradient chiếu sáng không đều, bóng tối và nếp nhăn giấy.",
            "image": encode_step_preview_b64(clamped)
        },
        {
            "step_number": 9,
            "title": "Lọc Hạt Bụi & Chấm Bẩn Quanh Chữ Qua Thành Phần Liên Thông (Connected Components)",
            "algorithm": "8-Connectivity Dust, Smudge & Isolated Inter-line Speck Filtering",
            "how": "Dò tìm các miền liên thông 8 hướng của hạt mực, loại bỏ các hạt bụi li ti, vết xám mờ sót quanh chữ và các chấm bẩn rơi cô lập giữa các dòng nhưng vẫn bảo toàn dấu chấm câu và chữ mảnh.",
            "why": "Làm sạch tinh khiết đến từng pixel, xử lý dứt điểm các hạt bụi và vết nhỏ còn sót lại xung quanh văn bản trước khi đưa sang OCR.",
            "image": encode_step_preview_b64(clean)
        }
    ]

    metrics = {
        "th_otsu": float(th_otsu),
        "th_paper": int(th_paper),
        "k_gauss": int(k),
        "sigma_gauss": round(float(sigma), 1),
        "white_ratio": round(white_ratio, 2),
        "connected_components_raw": int(num_cc_raw),
        "connected_components_clean": int(num_cc_clean),
        "version": "gaussian",
        "bg_label": f"Độ mờ σ = {sigma:.1f}"
    }
    return clean, steps_data, metrics

def process_pipeline_median(gray):
    """Bản 2: Phiên bản cải tiến dùng Adaptive Median Filter bảo toàn biên, giữ chữ đậm nét cho OCR & tự tách nền tối."""
    h, w = gray.shape[:2]
    k = max(13, int(min(h, w) * 0.045))
    k = min(k, 31)
    if k % 2 == 0:
        k += 1

    # Bước 2: Ước lượng nền bảo toàn biên bằng Median Filter
    if HAS_CV2:
        bg = cv2.medianBlur(gray, k)
    else:
        pil_img = Image.fromarray(gray)
        pil_bg = pil_img.filter(ImageFilter.MedianFilter(size=min(k, 15)))
        bg = np.array(pil_bg)

    # Bước 3: Phép chia trực tiếp trên nền giấy (không cộng offset làm nhạt chữ trong vùng ố) + tự động tách nền bàn tối
    bg_safe = np.maximum(bg.astype(np.float32), 1.0)
    div = np.clip((gray.astype(np.float32) / bg_safe) * 255.0, 0, 255).astype(np.uint8)

    dark_bg_mask = None
    if HAS_CV2:
        bg_p90 = float(np.percentile(bg, 90))
        bg_p15 = float(np.percentile(bg, 15))
        if bg_p90 > 130.0 and bg_p15 < bg_p90 * 0.42:
            bg_th, _ = cv2.threshold(bg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cand_paper = (bg >= int(bg_th * 0.85)).astype(np.uint8)
            se_p = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            cand_paper = cv2.morphologyEx(cand_paper, cv2.MORPH_CLOSE, se_p)
            num_p, lbl_p, stats_p, _ = cv2.connectedComponentsWithStats(cand_paper, 8)
            if num_p > 1:
                largest = 1 + int(np.argmax(stats_p[1:, cv2.CC_STAT_AREA]))
                if stats_p[largest, cv2.CC_STAT_AREA] > 0.20 * h * w:
                    paper_mask = (lbl_p == largest).astype(np.uint8)
                    contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(paper_mask, contours, -1, 1, thickness=cv2.FILLED)
                    se_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                    paper_mask = cv2.erode(paper_mask, se_e, iterations=1)
                    dark_bg_mask = (paper_mask == 0)

    div_work = div.copy()
    if dark_bg_mask is not None:
        div_work[dark_bg_mask] = 255

    # Bước 4: Otsu
    if HAS_CV2:
        th_otsu, _ = cv2.threshold(div_work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th_paper = int(th_otsu + (255 - th_otsu) * 0.45)
        otsu_bin = cv2.threshold(div_work, int(th_otsu), 255, cv2.THRESH_BINARY)[1]
    else:
        th_otsu = 181.0
        th_paper = 215
        otsu_bin = (div_work >= th_otsu).astype(np.uint8) * 255

    # Bước 5: Gamma 1.8 kéo đậm nét chữ trong vùng ố ngang bằng vùng sạch để OCR đọc chuẩn
    v = np.clip(div_work.astype(np.float32) / float(max(th_paper, 1)), 0.0, 1.0)
    contrast = np.clip(np.power(v, 1.8) * 255.0, 0, 255).astype(np.uint8)
    contrast[div_work >= th_paper] = 255

    # Bước 6: Adaptive Black-Hat trên ảnh phẳng nền
    bh_k = 5 if min(h, w) < 600 else 9
    if HAS_CV2:
        se_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh_k, bh_k))
        bh = cv2.morphologyEx(div_work, cv2.MORPH_BLACKHAT, se_bh)
        bh_visual = np.clip(bh.astype(np.float32) * 3.5, 0, 255).astype(np.uint8)
    else:
        bh = np.zeros_like(gray)
        bh_visual = np.zeros_like(gray)

    # Bước 7: Triệt tiêu quầng ố nhẹ nhàng (BH < 15) để không làm mờ nét chữ nghiêng mảnh
    stain_suppressed = contrast.copy()
    if HAS_CV2:
        stain_haze = (div_work >= int(th_otsu)) & (bh < 15)
        stain_suppressed[stain_haze] = 255

    # Bước 8: Ép nền giấy trắng tuyệt đối 100%
    clamped = stain_suppressed.copy()
    clamped[div_work >= th_paper] = 255

    # Bước 9: Lọc hạt bụi & chấm bẩn sót lại quanh chữ
    clean = filter_dust_and_isolated_specks(clamped)

    num_cc_raw, num_cc_clean = 0, 0
    if HAS_CV2:
        num_cc_raw, _ = cv2.connectedComponents((gray < 200).astype(np.uint8))
        num_cc_clean, _ = cv2.connectedComponents((clean < 200).astype(np.uint8))

    white_ratio = float(np.mean(clean == 255) * 100)

    steps_data = [
        {
            "step_number": 1,
            "title": "Chuyển Đổi Mức Xám (Grayscale Conversion)",
            "algorithm": "Y = 0.299*R + 0.587*G + 0.114*B",
            "how": "Bóc tách bức ảnh đa kênh màu (RGB) thành ma trận 1 kênh cường độ sáng từ 0 đến 255.",
            "why": "Loại bỏ nhiễu màu sắc của vết ố vàng/nâu, giảm 66% bộ nhớ và tăng tốc xử lý gấp 3 lần.",
            "image": encode_step_preview_b64(gray)
        },
        {
            "step_number": 2,
            "title": "Ước Lượng Trường Nền Bảo Toàn Biên (Adaptive Median Filter)",
            "algorithm": f"Median Filter 2D (k = {k}×{k})",
            "how": f"Dùng bộ lọc trung vị kích thước {k}×{k} để xóa nét chữ mảnh nhưng vẫn bám sát đường biên sắc của vết ố và nếp gấp giấy.",
            "why": "Khác với Gaussian Blur làm nhòe qua mép vết ố/nếp gấp, Median Filter bảo toàn đường biên sắc giúp bước chia nền khử vết ố sạch hơn đáng kể.",
            "image": encode_step_preview_b64(bg)
        },
        {
            "step_number": 3,
            "title": "Làm Phẳng Nền & Tự Động Tách Nền Tối (Background Division)",
            "algorithm": "I_div = clip((I_gray / I_bg) * 255, 0, 255)",
            "how": "Chia trực tiếp ảnh xám cho bản đồ nền Median để giữ nguyên độ đậm của chữ nằm trong vùng ố; đồng thời tự cô lập nền bàn tối bên ngoài trang giấy nếu có.",
            "why": "Kéo toàn bộ nền giấy bị ố về mức trắng đồng nhất trong khi vẫn giữ trọn độ đậm sâu của nét mực nằm ngay trên vùng ố cà phê.",
            "image": encode_step_preview_b64(div_work)
        },
        {
            "step_number": 4,
            "title": "Phân Tích Lược Đồ & Phân Ngưỡng Tự Động Otsu (Automatic Otsu Thresholding)",
            "algorithm": f"Otsu Inter-Class Variance -> T_otsu = {th_otsu:.1f}, T_paper = {th_paper}",
            "how": f"Tự động phân tích histogram của vùng giấy phẳng nền để tìm ngưỡng tối ưu T = {th_otsu:.1f} và ranh giới giấy trắng T_paper = {th_paper}.",
            "why": "Tự động thích ứng theo độ đậm nhạt của từng tài liệu mà không cần đặt ngưỡng cứng.",
            "image": encode_step_preview_b64(otsu_bin)
        },
        {
            "step_number": 5,
            "title": "Tăng Cường Tương Phản Phi Tuyến Nét Chữ (Non-linear Gamma Contrast)",
            "algorithm": f"v = clip(I_div / {th_paper}, 0, 1); I_gamma = (v ^ 1.8) * 255",
            "how": f"Chuẩn hóa ảnh phẳng nền theo T_paper={th_paper} về [0, 1] rồi áp dụng đường cong Gamma bậc 1.8.",
            "why": "Kéo đậm nét chữ nằm trong vùng ố ngang bằng với vùng giấy sạch để OCR nhận dạng chính xác mà viền chữ vẫn tròn mượt.",
            "image": encode_step_preview_b64(contrast)
        },
        {
            "step_number": 6,
            "title": "Biến Đổi Hình Thái Học Black-Hat Thích Ứng (Adaptive Black-Hat)",
            "algorithm": f"BH(I) = Closing(I_div, {bh_k}×{bh_k} Ellipse) - I_div",
            "how": f"Áp dụng phép Đóng hình thái học với phần tử elip {bh_k}×{bh_k} trên ảnh phẳng nền rồi trừ đi chính nó.",
            "why": "Tách riêng bản đồ nét mực thật sự trên nền tối, phân biệt rõ chữ mảnh với quầng ố và bóng nếp gấp.",
            "image": encode_step_preview_b64(bh_visual)
        },
        {
            "step_number": 7,
            "title": "Triệt Tiêu Quầng Sương Mờ Vết Ố & Bóng Nếp Gấp (Stain Suppression)",
            "algorithm": f"Haze = (I_div >= {int(th_otsu)}) AND (BH < 15) -> I[Haze] = 255",
            "how": "Kết hợp ngưỡng Otsu và bản đồ nét mực Black-Hat để tẩy trắng hoàn toàn các điểm ảnh không phải nét mực thật.",
            "why": "Xóa sạch quầng sương mờ quanh vết ố cà phê và bóng mờ nếp gấp mà vẫn giữ trọn nét chữ nghiêng mảnh.",
            "image": encode_step_preview_b64(stain_suppressed)
        },
        {
            "step_number": 8,
            "title": "Ép Nền Giấy Trắng Tuyệt Đối 100% (100% White Background Clamping)",
            "algorithm": f"I_clamped[I_div >= {th_paper}] = 255",
            "how": f"Gán cứng giá trị 255 cho toàn bộ điểm ảnh nền vượt qua ngưỡng ranh giới giấy T_paper = {th_paper}.",
            "why": "Đưa nền tài liệu về trắng số học 255 tuyệt đối, xóa sổ hoàn toàn gợn bóng chiếu sáng.",
            "image": encode_step_preview_b64(clamped)
        },
        {
            "step_number": 9,
            "title": "Lọc Hạt Bụi & Chấm Bẩn Quanh Chữ Qua Thành Phần Liên Thông (Connected Components)",
            "algorithm": "8-Connectivity Dust, Smudge & Isolated Inter-line Speck Filtering",
            "how": "Quét thành phần liên thông 8 hướng để xóa hạt bụi li ti, vết xám mờ quanh chữ và các chấm bẩn rơi cô lập giữa các dòng nhưng vẫn giữ nguyên dấu chấm câu và nét mảnh.",
            "why": "Dọn sạch các hạt bụi và vết nhỏ sót lại quanh văn bản, tối ưu chất lượng ảnh cho OCR.",
            "image": encode_step_preview_b64(clean)
        }
    ]

    metrics = {
        "th_otsu": float(th_otsu),
        "th_paper": int(th_paper),
        "k_gauss": int(k),
        "sigma_gauss": 0.0,
        "white_ratio": round(white_ratio, 2),
        "connected_components_raw": int(num_cc_raw),
        "connected_components_clean": int(num_cc_clean),
        "version": "median",
        "bg_label": "Median bảo toàn biên"
    }
    return clean, steps_data, metrics

def build_ocr_ready_image(gray, clean):
    """
    Tạo bản ảnh tối ưu riêng cho OCR (đặc biệt cho Tesseract.js trên Vercel khi ảnh là giấy nhàu/nghiêng trên nền tối):
    - Chỉ kích hoạt khi phát hiện tờ giấy nằm trên nền tối (như bàn gỗ).
    - Cô lập trang giấy, tách bóng nếp gấp xám mềm và xoay thẳng dòng chữ bằng Horizontal Projection Profile.
    - Không làm thay đổi ảnh kết quả clean_image hiển thị cho người dùng.
    """
    if not HAS_CV2:
        return None

    h, w = gray.shape[:2]
    k_bg = min(31, max(13, int(min(h, w) * 0.05) | 1))
    bg = cv2.medianBlur(gray, k_bg)
    bg_p90 = float(np.percentile(bg, 90))
    bg_p15 = float(np.percentile(bg, 15))
    if not (bg_p90 > 130.0 and bg_p15 < bg_p90 * 0.42):
        return None

    ocr_img = clean.copy()
    bg_th, _ = cv2.threshold(bg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cand = (bg >= int(bg_th * 0.85)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    num_p, lbl_p, stats_p, _ = cv2.connectedComponentsWithStats(cand, 8)
    if num_p > 1:
        largest = 1 + int(np.argmax(stats_p[1:, cv2.CC_STAT_AREA]))
        if stats_p[largest, cv2.CC_STAT_AREA] > 0.20 * h * w:
            pm = (lbl_p == largest).astype(np.uint8)
            cnts, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(pm, cnts, -1, 1, thickness=cv2.FILLED)
            pm = cv2.erode(pm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
            ocr_img[pm == 0] = 255

    # Loại bỏ bóng nếp gấp xám mềm, chỉ giữ lại lõi nét chữ đen đậm cho Tesseract.js
    dark_core = (ocr_img < 75).astype(np.uint8)
    if int(np.sum(dark_core)) > 80:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        halo = cv2.dilate(dark_core, se, iterations=1)
        ocr_img[(ocr_img >= 95) & (halo == 0)] = 255
        mask_ink = (ocr_img < 220).astype(np.uint8)
        n_c, l_c, st_c, _ = cv2.connectedComponentsWithStats(mask_ink, 8)
        for i in range(1, n_c):
            area = st_c[i, cv2.CC_STAT_AREA]
            x, y, bw, bh_b = st_c[i, cv2.CC_STAT_LEFT], st_c[i, cv2.CC_STAT_TOP], st_c[i, cv2.CC_STAT_WIDTH], st_c[i, cv2.CC_STAT_HEIGHT]
            rm = (l_c[y:y+bh_b, x:x+bw] == i)
            vals = ocr_img[y:y+bh_b, x:x+bw][rm]
            if area <= 6 or float(np.percentile(vals, 15)) > 90 or float(np.mean(vals)) > 145:
                ocr_img[y:y+bh_b, x:x+bw][rm] = 255

    # Auto-Deskew bằng Horizontal Projection Profile (-25 đến +25 độ)
    small_w = min(w, 320)
    scale = small_w / float(w)
    small_bin = (cv2.resize(ocr_img, (small_w, max(1, int(h * scale)))) < 150).astype(np.float32)
    sh, sw = small_bin.shape
    best_ang, best_score = 0.0, -1.0
    for ang in range(-25, 26, 1):
        M = cv2.getRotationMatrix2D((sw / 2.0, sh / 2.0), ang, 1.0)
        rot = cv2.warpAffine(small_bin, M, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
        score = float(np.var(np.sum(rot, axis=1)))
        if score > best_score:
            best_score = score
            best_ang = float(ang)

    if abs(best_ang) >= 2.0:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), best_ang, 1.0)
        ocr_img = cv2.warpAffine(ocr_img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)

    return ocr_img

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
async def denoise(
    file: UploadFile = File(...),
    run_ocr: bool = Form(True),
    version: str = Form("gaussian")
):
    try:
        contents = await file.read()
        if not contents:
            return JSONResponse({"error": "File ảnh rỗng"}, status_code=400)

        # Giải mã ảnh & giới hạn kích thước tối đa 1400px để tránh quá tải bộ nhớ / payload Vercel
        if HAS_CV2:
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return JSONResponse({"error": "Không thể giải mã file ảnh"}, status_code=400)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            if max(h, w) > 1400:
                scale = 1400.0 / max(h, w)
                gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        else:
            pil_img = Image.open(io.BytesIO(contents)).convert('L')
            w, h = pil_img.size
            if max(h, w) > 1400:
                scale = 1400.0 / max(h, w)
                pil_img = pil_img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
            gray = np.array(pil_img)

        # Chọn phiên bản thuật toán: gaussian (mặc định - quy trình gốc Word/PPT) hoặc median (cải tiến bảo toàn biên)
        if version == "median":
            clean, steps, metrics = process_pipeline_median(gray)
        else:
            clean, steps, metrics = process_pipeline_gaussian(gray)

        # Chuẩn bị bản ảnh tối ưu OCR (tự tách bàn tối + xoay thẳng nếu là ảnh chụp tờ giấy nhàu nghiêng)
        ocr_ready = build_ocr_ready_image(gray, clean) if run_ocr else None
        ocr_target = ocr_ready if ocr_ready is not None else clean

        # Chạy OCR tiếng Anh (nếu server có RapidOCR/pytesseract)
        ocr_clean_lines = run_ocr_text(ocr_target) if run_ocr else []
        ocr_raw_lines = run_ocr_text(gray) if run_ocr else []

        raw_b64 = encode_img_b64(gray)
        clean_b64 = encode_img_b64(clean)

        return {
            "success": True,
            "filename": file.filename or "document.png",
            "width": int(clean.shape[1]),
            "height": int(clean.shape[0]),
            "raw_image": f"data:image/png;base64,{raw_b64}",
            "clean_image": f"data:image/png;base64,{clean_b64}",
            "ocr_image": encode_step_preview_b64(ocr_ready, max_width=1200, quality=92) if ocr_ready is not None else None,
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
