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

def restore_crisp_document(gray):
    """
    Thuật toán Xử lý ảnh số (DIP) cổ điển - Nhóm 3:
    1. Adaptive Gaussian Background Estimation: Co giãn hạt nhân theo độ phân giải ảnh
    2. Background Division: Triệt tiêu vết ố loang, bóng mờ, đưa nền về 255
    3. Non-linear Contrast Transfer Curve: Số mũ gamma=1.4 giúp chữ đen đậm
    4. Paper Background Clamping: Ép nền giấy trắng tinh khiết tuyệt đối 255
    """
    h, w = gray.shape[:2]
    k = max(71, int(min(h, w) * 0.15))
    if k % 2 == 0:
        k += 1
    sigma = max(30.0, float(k) / 2.5)

    if HAS_CV2:
        bg = cv2.GaussianBlur(gray, (k, k), sigma)
        div = cv2.divide(gray, bg, scale=255)
    else:
        # Fallback dùng Pillow GaussianFilter nếu môi trường không có OpenCV
        pil_img = Image.fromarray(gray)
        pil_bg = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
        bg_arr = np.array(pil_bg).astype(float) + 1e-5
        div = np.clip((gray.astype(float) / bg_arr) * 255.0, 0, 255).astype(np.uint8)

    v = np.clip(div.astype(float) / 225.0, 0.0, 1.0)
    v_crisp = np.power(v, 1.4) * 255.0
    out = np.clip(v_crisp, 0, 255).astype(np.uint8)
    out[div >= 225] = 255
    return out

@app.get("/")
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
            clean = restore_crisp_document(gray)

            _, buffer = cv2.imencode('.png', clean)
            clean_b64 = base64.b64encode(buffer).decode('utf-8')

            _, raw_buf = cv2.imencode('.png', gray)
            raw_b64 = base64.b64encode(raw_buf).decode('utf-8')
        else:
            pil_img = Image.open(io.BytesIO(contents)).convert('L')
            gray = np.array(pil_img)
            clean = restore_crisp_document(gray)

            # Encode PNG
            clean_pil = Image.fromarray(clean)
            buf = io.BytesIO()
            clean_pil.save(buf, format="PNG")
            clean_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            raw_buf = io.BytesIO()
            pil_img.save(raw_buf, format="PNG")
            raw_b64 = base64.b64encode(raw_buf.getvalue()).decode('utf-8')

        return {
            "success": True,
            "filename": file.filename or "document.png",
            "width": int(clean.shape[1]),
            "height": int(clean.shape[0]),
            "raw_image": f"data:image/png;base64,{raw_b64}",
            "clean_image": f"data:image/png;base64,{clean_b64}",
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
