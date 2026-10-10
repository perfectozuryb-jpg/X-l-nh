import io
import os
import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
    1. Adaptive Gaussian Background Estimation (tự thích ứng theo kích thước ảnh)
    2. Background Division (triệt tiêu gradient ánh sáng, vết ố loang, nếp nhăn)
    3. Non-linear Contrast Transfer Curve (gamma=1.4 giúp chữ đen đậm)
    4. Paper Background Clamping (nền trắng tuyệt đối 255)
    """
    h, w = gray.shape[:2]
    # Tự động co giãn kích thước kernel theo độ phân giải tài liệu
    k = max(71, int(min(h, w) * 0.15))
    if k % 2 == 0:
        k += 1
    sigma = max(30.0, float(k) / 2.5)

    bg = cv2.GaussianBlur(gray, (k, k), sigma)
    div = cv2.divide(gray, bg, scale=255)

    v = np.clip(div.astype(float) / 225.0, 0.0, 1.0)
    v_crisp = np.power(v, 1.4) * 255.0
    out = np.clip(v_crisp, 0, 255).astype(np.uint8)
    out[div >= 225] = 255
    return out


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Denoising Dirty Documents DIP Pipeline"}

@app.post("/api/denoise")
async def denoise(file: UploadFile = File(...), run_ocr: bool = Form(False)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"error": "Không thể giải mã file ảnh"}, status_code=400)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clean = restore_crisp_document(gray)

        # Encode ảnh sau xử lý thành PNG base64
        _, buffer = cv2.imencode('.png', clean)
        clean_b64 = base64.b64encode(buffer).decode('utf-8')

        # Encode ảnh gốc thành PNG base64 để hiển thị
        _, raw_buf = cv2.imencode('.png', gray)
        raw_b64 = base64.b64encode(raw_buf).decode('utf-8')

        ocr_result = []
        if run_ocr:
            try:
                from rapidocr_onnxruntime import RapidOCR
                engine = RapidOCR()
                res, _ = engine(clean)
                if res:
                    for box, text, score in res:
                        ocr_result.append({"text": text, "confidence": round(float(score), 2)})
            except Exception as e:
                ocr_result.append({"text": f"(Lỗi OCR: {str(e)})", "confidence": 0.0})

        return {
            "success": True,
            "filename": file.filename,
            "width": int(clean.shape[1]),
            "height": int(clean.shape[0]),
            "raw_image": f"data:image/png;base64,{raw_b64}",
            "clean_image": f"data:image/png;base64,{clean_b64}",
            "ocr_result": ocr_result
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Phục vụ giao diện web public khi chạy thử trên máy local
if os.path.exists("public"):
    app.mount("/", StaticFiles(directory="public", html=True), name="static")
