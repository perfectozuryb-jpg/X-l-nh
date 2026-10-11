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

app = FastAPI(title="Denoising Dirty Documents API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def encode_img_b64(arr):
    """Helper chuyển mảng numpy sang chuỗi base64 PNG."""
    if HAS_CV2:
        _, buf = cv2.imencode('.png', arr)
        return base64.b64encode(buf).decode('utf-8')
    else:
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode('utf-8')

def process_pipeline_with_steps(gray):
    """
    Quy trình Xử lý ảnh số (DIP) 5 bước phân rã chi tiết - Nhóm 3:
    Trả về ảnh kết quả cuối cùng kèm ảnh trung gian và giải thích khoa học của từng bước.
    """
    h, w = gray.shape[:2]
    k = max(71, int(min(h, w) * 0.15))
    if k % 2 == 0:
        k += 1
    sigma = max(30.0, float(k) / 2.5)

    # Bước 2: Ước lượng trường nền
    if HAS_CV2:
        bg = cv2.GaussianBlur(gray, (k, k), sigma)
        div = cv2.divide(gray, bg, scale=255)
    else:
        pil_img = Image.fromarray(gray)
        pil_bg = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
        bg = np.array(pil_bg)
        bg_arr = bg.astype(float) + 1e-5
        div = np.clip((gray.astype(float) / bg_arr) * 255.0, 0, 255).astype(np.uint8)

    # Bước 4: Tăng tương phản phi tuyến
    v = np.clip(div.astype(float) / 225.0, 0.0, 1.0)
    contrast = np.clip(np.power(v, 1.4) * 255.0, 0, 255).astype(np.uint8)

    # Bước 5: Ép nền trắng 100%
    clean = contrast.copy()
    clean[div >= 225] = 255

    steps_data = [
        {
            "step_number": 1,
            "title": "Chuyển Đổi Mức Xám (Grayscale Conversion)",
            "algorithm": "Y = 0.299*R + 0.587*G + 0.114*B",
            "how": "Bóc tách bức ảnh đa kênh màu (RGB) thành ma trận 1 kênh cường độ sáng từ 0 (đen) đến 255 (trắng).",
            "why": "Loại bỏ thông tin màu sắc gây nhiễu, tập trung vào cường độ phản xạ ánh sáng của mực in và nền giấy. Giúp giảm 66% bộ nhớ và tăng tốc độ xử lý ma trận gấp 3 lần.",
            "image": f"data:image/png;base64,{encode_img_b64(gray)}"
        },
        {
            "step_number": 2,
            "title": "Ước Lượng Trường Nền (Adaptive Gaussian Blur)",
            "algorithm": f"Gaussian Blur 2D (k = {k}×{k}, σ = {sigma:.1f})",
            "how": f"Tích chập ảnh mức xám với bộ lọc Gauss có kích thước hạt nhân siêu lớn ({k}×{k}) tự động co giãn theo 15% kích thước tài liệu.",
            "why": "Nét chữ là chi tiết biến thiên tần số cao (kích thước nhỏ vài pixel), trong khi vết ố loang, bóng đổ và nếp gấp là thành phần tần số thấp (trải rộng toàn trang). Hạt nhân Gauss lớn sẽ làm mờ hoàn toàn nét chữ, chỉ giữ lại mô hình biến thiên của nền và vết ố để bóc tách.",
            "image": f"data:image/png;base64,{encode_img_b64(bg)}"
        },
        {
            "step_number": 3,
            "title": "Làm Phẳng Nền Bằng Phép Chia (Background Division)",
            "algorithm": "I_div = (I_gray / I_bg) * 255",
            "how": "Lấy ma trận điểm ảnh xám chia theo từng phần tử cho ma trận trường nền, sau đó nhân với 255.",
            "why": "Tại mọi vị trí nền (kể cả vùng bị ố cà phê đậm hay bóng tối), giá trị gốc xấp xỉ giá trị nền (I_gray ≈ I_bg) nên tỉ số xấp xỉ 1.0, nhân 255 sẽ trở thành nền trắng phẳng lì 255! Ngược lại, tại nét chữ I_gray << I_bg nên tỉ số rất nhỏ, bảo toàn được độ đậm của nét mực.",
            "image": f"data:image/png;base64,{encode_img_b64(div)}"
        },
        {
            "step_number": 4,
            "title": "Tăng Tương Phản Phi Tuyến Nét Chữ (Gamma = 1.4)",
            "algorithm": "I_crisp = ((I_div / 225) ^ 1.4) * 255",
            "how": "Áp dụng hàm biến đổi phi tuyến với số mũ gamma = 1.4 lên dải giá trị chuẩn hóa của ảnh.",
            "why": "Sau phép chia nền, nét chữ có thể bị hơi xám hoặc mờ nhạt. Thay vì phân ngưỡng nhị phân cứng (Hard Thresholding) gây răng cưa đứt nét, hàm phi tuyến kéo hạt mực in về sắc đen sâu (0 - 30), đồng thời giữ lại hiệu ứng khử răng cưa (Antialiasing) ở đường viền ký tự giúp chữ tròn trịa, sắc nét.",
            "image": f"data:image/png;base64,{encode_img_b64(contrast)}"
        },
        {
            "step_number": 5,
            "title": "Triệt Tiêu Vết Ố & Ép Nền Trắng 100% (Paper Clamping)",
            "algorithm": "I_clean[I_div >= 225] = 255",
            "how": "Tất cả điểm ảnh có giá trị chuẩn hóa từ 225 trở lên (thuộc nền giấy, vi vết ố nhạt hoặc nếp gấp đã mờ) được ép dứt điểm về 255.",
            "why": "Đảm bảo nền giấy đạt độ trắng tinh khiết tuyệt đối 100% (255), triệt tiêu sạch sẽ các đốm hạt lấm chấm còn sót lại, giúp ảnh trong trẻo như tài liệu mới in và tối ưu hóa tối đa cho công cụ nhận dạng chữ OCR.",
            "image": f"data:image/png;base64,{encode_img_b64(clean)}"
        }
    ]

    return clean, steps_data

@app.get("/")
def root():
    if os.path.exists("public/index.html"):
        from fastapi.responses import FileResponse
        return FileResponse("public/index.html")
    return {
        "status": "ok",
        "service": "Denoising Dirty Documents DIP Pipeline"
    }

@app.get("/api")
@app.get("/api/health")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Denoising Dirty Documents DIP Pipeline",
        "engine": "OpenCV" if HAS_CV2 else "Pillow-NumPy"
    }


@app.post("/api/denoise")
@app.post("/denoise")
async def denoise(file: UploadFile = File(...), run_ocr: bool = Form(False)):
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
            clean, steps = process_pipeline_with_steps(gray)

            raw_b64 = encode_img_b64(gray)
            clean_b64 = encode_img_b64(clean)
        else:
            pil_img = Image.open(io.BytesIO(contents)).convert('L')
            gray = np.array(pil_img)
            clean, steps = process_pipeline_with_steps(gray)

            raw_b64 = encode_img_b64(gray)
            clean_b64 = encode_img_b64(clean)

        return {
            "success": True,
            "filename": file.filename or "document.png",
            "width": int(clean.shape[1]),
            "height": int(clean.shape[0]),
            "raw_image": f"data:image/png;base64,{raw_b64}",
            "clean_image": f"data:image/png;base64,{clean_b64}",
            "steps": steps,
            "ocr_result": []
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Phục vụ giao diện tĩnh khi chạy thử trên máy local (không chạy trên Vercel CDN)
if os.path.exists("public") and not os.environ.get("VERCEL"):
    try:
        app.mount("/", StaticFiles(directory="public", html=True), name="static")
    except Exception:
        pass
