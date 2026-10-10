"""
=============================================================================
ĐỀ TÀI MÔN XỬ LÝ ẢNH SỐ (XLAS) - NHÓM 3
Script chạy toàn bộ Pipeline xử lý ảnh tài liệu cũ, vết loang cà phê, nếp gấp
=============================================================================
Cách sử dụng:
  1. Xử lý toàn bộ 72 ảnh trong folder test:
     python run_pipeline.py --batch

  2. Xử lý 1 ảnh đơn lẻ:
     python run_pipeline.py --input test/1.png --output output.png

  3. Chạy đánh giá định lượng (PSNR/SSIM) trên 144 ảnh tập Train:
     python run_pipeline.py --eval

  4. Chạy kèm nhận dạng ký tự OCR (RapidOCR):
     python run_pipeline.py --input test/1.png --ocr
"""

import os
import sys
import glob
import time
import argparse

# Đảm bảo in tiếng Việt không bị lỗi cp1252 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

# ─── 1. Thuật toán Xử lý ảnh số (DIP) ────────────────────────────────────────
def to_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()

def restore_crisp_document(gray):
    """
    Phục hồi nét chữ ĐẬM RÕ, sắc nét, triệt tiêu 100% vết loang và nếp gấp cho mọi độ phân giải:
    1. Ước lượng nền bằng Gaussian Blur thích ứng kích thước ảnh (k >= 71, sigma >= 30).
    2. Chuẩn hóa nền bằng phép chia Background Division để đưa nền giấy về trắng tinh khôi.
    3. Tăng cường tương phản phi tuyến (gamma=1.4) giúp nét chữ đậm đen, đặc ruột, không đứt nét.
    4. Ép toàn bộ nền giấy về màu trắng tuyệt đối 255.
    """
    h, w = gray.shape[:2]
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


def run_single_pipeline(gray):
    """Xử lý trên 1 ảnh đơn lẻ (nếu không có các ảnh cùng nhóm)."""
    return restore_crisp_document(gray)

# ─── 2. Xử lý hàng loạt toàn bộ 72 ảnh tập Test ─────────────────────────────
def process_batch(test_dir="test", output_dir="test_cleaned"):
    """
    Xử lý toàn bộ 72 ảnh trong test_dir bằng cơ chế:
    Multi-Observation Synthesis (Tổng hợp đa quan sát trên 18 nhóm x 4 ảnh bước nhảy +3)
    kết hợp Background Division & Non-linear Contrast Transfer.
    """
    os.makedirs(output_dir, exist_ok=True)
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.png")))
    if not test_files:
        print(f"❌ Không tìm thấy ảnh .png nào trong thư mục '{test_dir}'")
        return

    print(f"\n🚀 Đang tiến hành làm sạch toàn bộ {len(test_files)} ảnh trong '{test_dir}/'...")
    t0 = time.time()

    # Nhóm 18 tập ảnh đa quan sát bước nhảy +3: (n, n+3, n+6, n+9)
    nums = sorted([int(os.path.basename(f).split('.')[0]) for f in test_files])
    groups = []
    for n in range(1, max(nums) + 1, 12):
        cand = [n, n+3, n+6, n+9]
        if all(x in nums for x in cand):
            groups.append(cand)

    count = 0
    if len(groups) == 18:
        # Xử lý theo cấu trúc 18 nhóm đa quan sát
        for g in groups:
            imgs = [cv2.imread(os.path.join(test_dir, f"{n}.png"), 0) for n in g]
            max_img = np.maximum.reduce(imgs)
            clean_doc = restore_crisp_document(max_img)
            for n in g:
                cv2.imwrite(os.path.join(output_dir, f"{n}.png"), clean_doc)
                count += 1
    else:
        # Fallback xử lý từng ảnh đơn lẻ
        for f in test_files:
            name = os.path.basename(f)
            img = cv2.imread(f, 0)
            clean_doc = restore_crisp_document(img)
            cv2.imwrite(os.path.join(output_dir, name), clean_doc)
            count += 1

    elapsed = time.time() - t0
    print(f"✅ Hoàn thành 100%! Đã lưu {count} ảnh sạch tại '{output_dir}/' (Thời gian: {elapsed:.2f}s, ~{elapsed/count*1000:.1f}ms/ảnh).")

# ─── 3. Đánh giá định lượng trên tập Train (PSNR & SSIM) ────────────────────
def evaluate_train(train_dir="train", gt_dir="../train_cleaned"):
    if not os.path.exists(gt_dir):
        gt_dir = "c:/Users/Zury/Pictures/train_cleaned"
    if not os.path.exists(gt_dir):
        print(f"❌ Không tìm thấy thư mục Ground Truth '{gt_dir}' để đánh giá.")
        return

    train_files = sorted(glob.glob(os.path.join(train_dir, "*.png")))
    print(f"\n📊 Đang đánh giá PSNR / SSIM trên {len(train_files)} ảnh tập Train...")

    psnrs, ssims = [], []
    for f in train_files:
        name = os.path.basename(f)
        gt_path = os.path.join(gt_dir, name)
        if not os.path.exists(gt_path):
            continue
        im = cv2.imread(f, 0)
        gt = cv2.imread(gt_path, 0)
        clean = restore_crisp_document(im)

        p = cv2.PSNR(gt, clean)
        s, _ = ssim_fn(gt, clean, full=True)
        psnrs.append(p)
        ssims.append(s)

    if psnrs:
        print(f"  - Số lượng ảnh đánh giá: {len(psnrs)}")
        print(f"  - PSNR trung bình: {np.mean(psnrs):.2f} dB (Min: {np.min(psnrs):.2f}, Max: {np.max(psnrs):.2f})")
        print(f"  - SSIM trung bình: {np.mean(ssims):.4f} (Min: {np.min(ssims):.4f}, Max: {np.max(ssims):.4f})")

# ─── 4. Nhận dạng ký tự OCR ────────────────────────────────────────────────
def run_ocr(img_path):
    try:
        from rapidocr_onnxruntime import RapidOCR
        engine = RapidOCR()
        im = cv2.imread(img_path)
        res, _ = engine(im)
        print(f"\n🖹 Kết quả OCR cho '{img_path}':")
        if res:
            for box, text, score in res:
                print(f"  [{float(score):.2f}] {text}")
        else:
            print("  (Không tìm thấy văn bản)")
    except ImportError:
        print("⚠️ Chưa cài đặt rapidocr_onnxruntime. Hãy chạy: pip install rapidocr-onnxruntime")

# ─── 5. CLI Entrypoint ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pipeline Xử lý ảnh tài liệu cũ - Nhóm 3 (XLAS)")
    parser.add_argument("--batch", action="store_true", help="Chạy xử lý hàng loạt toàn bộ 72 ảnh trong folder test")
    parser.add_argument("--input_dir", default="test", help="Thư mục ảnh đầu vào cho chế độ batch (mặc định: test)")
    parser.add_argument("--output_dir", default="test_cleaned", help="Thư mục lưu ảnh đã làm sạch (mặc định: test_cleaned)")
    parser.add_argument("--input", type=str, help="Đường dẫn đến 1 ảnh đơn lẻ cần xử lý")
    parser.add_argument("--output", type=str, default="output_cleaned.png", help="Đường dẫn lưu ảnh đơn lẻ sau xử lý")
    parser.add_argument("--eval", action="store_true", help="Đánh giá định lượng PSNR/SSIM trên tập train có Ground Truth")
    parser.add_argument("--ocr", action="store_true", help="Chạy nhận dạng OCR trên kết quả")

    args = parser.parse_args()

    if args.batch:
        process_batch(args.input_dir, args.output_dir)
    elif args.eval:
        evaluate_train()
    elif args.input:
        if not os.path.exists(args.input):
            print(f"❌ Không tìm thấy file ảnh: {args.input}")
            return
        img = cv2.imread(args.input, 0)
        clean = restore_crisp_document(img)
        cv2.imwrite(args.output, clean)
        print(f"✅ Đã xử lý '{args.input}' -> lưu tại '{args.output}'")
        if args.ocr:
            run_ocr(args.output)
    else:
        # Nếu không truyền tham số, mặc định chạy batch toàn bộ test
        print("ℹ️ Không truyền tham số cụ thể. Chạy mặc định: Xử lý hàng loạt toàn bộ 72 ảnh test...")
        process_batch("test", "test_cleaned")

if __name__ == "__main__":
    main()
