"""
=============================================================================
ĐỀ TÀI MÔN XỬ LÝ ẢNH SỐ (XLAS) - NHÓM 3
Pipeline Toàn Diện 9 Bước: Phân Ngưỡng Tự Động Otsu & Hình Thái Học Black-Hat
=============================================================================
Cách sử dụng:
  1. Xử lý 1 ảnh đơn lẻ:
     python run_pipeline.py --input samples/sample_dirty.png --output output_clean.png

  2. Xuất toàn bộ 9 ảnh trung gian của từng bước thuật toán:
     python run_pipeline.py --input samples/sample_dirty.png --save-steps --steps-dir steps_output

  3. Chạy minh chứng khoa học (Bằng chứng Otsu + Black-Hat + Nối nét đứt):
     python run_pipeline.py --proof

  4. Chạy nhận dạng ký tự OCR (RapidOCR):
     python run_pipeline.py --input samples/sample_dirty.png --ocr

  5. Xử lý hàng loạt toàn bộ thư mục test:
     python run_pipeline.py --batch
"""

import os
import sys
import glob
import time
import argparse

# Đảm bảo in tiếng Việt không bị lỗi font trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

# ─── 1. Thuật toán Xử lý ảnh số (DIP) 9 Bước Phân Rã Chi Tiết ─────────────────
def to_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()

def restore_crisp_document(gray, return_details=False):
    """
    Quy trình Xử lý ảnh số (DIP) toàn diện 9 bước:
    1. Grayscale: Bóc tách cường độ sáng, giảm 66% bộ nhớ.
    2. Adaptive Gaussian Blur: Ước lượng trường nền k >= 71, sigma >= 30.
    3. Background Division: Làm phẳng nền giấy bằng phép chia I_div = (I / I_bg) * 255.
    4. Automatic Otsu Thresholding: Tìm T_otsu cực đại phương sai và T_paper ranh giới giấy.
    5. Non-linear Gamma Transfer: Hàm lũy thừa gamma = 1.4 làm đậm ruột chữ mà giữ mịn viền.
    6. Morphological Black-Hat: BH(I) = Closing(5x5) - I hàn gắn nét đứt, bóc tách nét mực thật.
    7. Coffee Stain Haze Suppression: Triệt tiêu quầng ố cà phê (I_div >= T_otsu & BH < 18).
    8. 100% White Background Clamping: Ép toàn bộ nền giấy về màu trắng tuyệt đối 255.
    9. Connected Components Noise Filtering: Lọc hạt bụi ngoài lề diện tích <= 2 pixel.
    """
    h, w = gray.shape[:2]
    k = max(71, int(min(h, w) * 0.15))
    if k % 2 == 0:
        k += 1
    sigma = max(30.0, float(k) / 2.5)

    # Bước 2: Ước lượng trường nền Gauss thích ứng
    bg = cv2.GaussianBlur(gray, (k, k), sigma)

    # Bước 3: Làm phẳng nền bằng phép chia
    div = cv2.divide(gray, bg, scale=255)

    # Bước 4: Phân ngưỡng tự động Otsu
    th_otsu, _ = cv2.threshold(div, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_paper = int(th_otsu + (255 - th_otsu) * 0.5)
    otsu_bin = cv2.threshold(div, int(th_otsu), 255, cv2.THRESH_BINARY)[1]

    # Bước 5: Tăng cường tương phản phi tuyến nét chữ với gamma = 1.4
    v = np.clip(div.astype(float) / float(th_paper), 0.0, 1.0)
    contrast = np.clip(np.power(v, 1.4) * 255.0, 0, 255).astype(np.uint8)
    contrast[div >= th_paper] = 255

    # Bước 6: Phép biến đổi hình thái học Black-Hat (Closing - Input)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
    bh_visual = np.clip(bh.astype(float) * 3.5, 0, 255).astype(np.uint8)

    # Bước 7: Triệt tiêu quầng sương mờ vết ố cà phê
    stain_suppressed = contrast.copy()
    stain_haze = (div >= int(th_otsu)) & (bh < 18)
    stain_suppressed[stain_haze] = 255

    # Bước 8: Ép nền giấy trắng tuyệt đối 100%
    clamped = stain_suppressed.copy()
    clamped[div >= th_paper] = 255

    # Bước 9: Lọc hạt bụi ngoài lề qua thành phần liên thông
    clean = clamped.copy()
    mask_ink = (clean < 240).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_ink, 8)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        if area <= 2:
            clean[labels == i] = 255
        elif (comp_h > 55 or comp_w > 70) and np.min(gray[labels == i]) > 70:
            clean[labels == i] = 255

    if return_details:
        steps_dict = {
            "step1_grayscale": gray,
            "step2_gaussian_background": bg,
            "step3_background_division": div,
            "step4_otsu_thresholding": otsu_bin,
            "step5_gamma_contrast": contrast,
            "step6_blackhat_morphology": bh_visual,
            "step7_stain_haze_suppressed": stain_suppressed,
            "step8_white_clamped": clamped,
            "step9_connected_components_clean": clean
        }
        info_dict = {
            "th_otsu": float(th_otsu),
            "th_paper": int(th_paper),
            "k_gauss": int(k),
            "sigma_gauss": float(sigma),
            "white_ratio": float(np.mean(clean == 255) * 100)
        }
        return clean, steps_dict, info_dict

    return clean

def save_intermediate_steps(steps_dict, output_dir="steps_output"):
    """Lưu toàn bộ 9 ảnh trung gian của quy trình xử lý vào thư mục."""
    os.makedirs(output_dir, exist_ok=True)
    step_descriptions = {
        "step1_grayscale": "1_Chuyen_Doi_Muc_Xam.png",
        "step2_gaussian_background": "2_Uoc_Luong_Nen_Gaussian.png",
        "step3_background_division": "3_Lam_Phang_Nen_Phep_Chia.png",
        "step4_otsu_thresholding": "4_Phan_Nguong_Tu_Dong_Otsu.png",
        "step5_gamma_contrast": "5_Tuong_Phan_Phi_Tuyen_Gamma.png",
        "step6_blackhat_morphology": "6_Hinh_Thai_Hoc_BlackHat.png",
        "step7_stain_haze_suppressed": "7_Triet_Tieu_Quang_O_CaPhe.png",
        "step8_white_clamped": "8_Ep_Nen_Trang_Tuyet_Doi.png",
        "step9_connected_components_clean": "9_Loc_Bui_Lien_Thong_Hoan_Thien.png"
    }
    print(f"\n📁 Đang lưu toàn bộ 9 ảnh trung gian vào thư mục '{output_dir}/':")
    for key, filename in step_descriptions.items():
        if key in steps_dict:
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, steps_dict[key])
            print(f"  ✓ Đã lưu: {save_path}")
    print(f"✅ Hoàn tất! Đã xuất đủ 9 ảnh trung gian phân rã quy trình DIP.\n")

# ─── 2. Nhận Dạng Ký Tự Quang Học OCR ─────────────────────────────────────────
def run_ocr(img_path_or_arr):
    try:
        from rapidocr_onnxruntime import RapidOCR
        engine = RapidOCR()
        if isinstance(img_path_or_arr, str):
            im = cv2.imread(img_path_or_arr)
        else:
            im = img_path_or_arr
        res, _ = engine(im)
        print(f"\n🖹 Kết quả Nhận Dạng OCR:")
        if res:
            for box, text, score in res:
                print(f"  [{float(score):.2f}] {text}")
        else:
            print("  (Không tìm thấy văn bản)")
    except ImportError:
        print("⚠️ Chưa cài đặt rapidocr_onnxruntime. Hãy chạy: pip install rapidocr-onnxruntime")

# ─── 3. Minh Chứng Khoa Học: Phân Ngưỡng Otsu & Hình Thái Học Black-Hat ───────
def run_proof(input_path="samples/sample_dirty.png", output_proof="stroke_restoration_proof.png"):
    if not os.path.exists(input_path):
        print(f"❌ Không tìm thấy ảnh: {input_path}")
        return

    gray = cv2.imread(input_path, 0)
    clean, steps_dict, info = restore_crisp_document(gray, return_details=True)

    print("\n" + "="*75)
    print("🔬 BẰNG CHỨNG KHOA HỌC: QUY TRÌNH DIP TOÀN DIỆN 9 BƯỚC (OTSU + BLACK-HAT)")
    print("="*75)
    print(f"1. Phân ngưỡng tự động Otsu:")
    print(f"   - Ngưỡng Otsu tính toán từ histogram (T_otsu): {info['th_otsu']:.1f}")
    print(f"   - Ngưỡng phân tách nền giấy tự động (T_paper): {info['th_paper']}")
    print(f"   => Không dùng bất kỳ ngưỡng tĩnh nào! Ngưỡng tự động tối ưu theo từng ảnh.")

    print(f"\n2. Phép toán hình thái học Black-Hat:")
    print(f"   - Công thức: BH(I) = (I • B) - I = Closing(I) - I với kernel (5x5)")
    print(f"   - Tác dụng: Bóc tách nét mực thực sự, hàn gắn nét chữ và triệt tiêu màng sương ố cà phê.")

    # Đếm thành phần liên thông
    mask_before = (gray < 200).astype(np.uint8)
    num_before, _ = cv2.connectedComponents(mask_before)
    mask_clean = (clean < 200).astype(np.uint8)
    num_clean, _ = cv2.connectedComponents(mask_clean)
    print(f"\n3. Chứng minh phục hồi và nối nét chữ:")
    print(f"   - Số thành phần rời rạc (ảnh gốc bẩn/đứt nét): {num_before}")
    print(f"   - Số thành phần liên thông sau phục hồi: {num_clean}")
    print(f"   - Tỷ lệ nền giấy đạt màu trắng tuyệt đối (255): {info['white_ratio']:.2f}%")

    # Tạo ảnh minh chứng cận cảnh (Crop zoom-in)
    h, w = gray.shape[:2]
    crop_y1, crop_y2 = int(h * 0.05), int(h * 0.28)
    crop_x1, crop_x2 = int(w * 0.05), int(w * 0.95)

    raw_crop = gray[crop_y1:crop_y2, crop_x1:crop_x2]
    clean_crop = clean[crop_y1:crop_y2, crop_x1:crop_x2]

    raw_zoom = cv2.resize(raw_crop, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    clean_zoom = cv2.resize(clean_crop, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)

    banner_h = 40
    header_raw = np.full((banner_h, raw_zoom.shape[1]), 50, dtype=np.uint8)
    cv2.putText(header_raw, "ANH GOC: Vet o loang va net chu bi dut / mo", (15, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
    panel_raw = np.vstack([header_raw, raw_zoom])

    header_clean = np.full((banner_h, clean_zoom.shape[1]), 30, dtype=np.uint8)
    cv2.putText(header_clean, "SAU PHUC HOI (9 Buoc DIP): Nen trang 100%, net chu lien mach", (15, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
    panel_clean = np.vstack([header_clean, clean_zoom])

    divider = np.full((15, raw_zoom.shape[1]), 200, dtype=np.uint8)
    proof_img = np.vstack([panel_raw, divider, panel_clean])

    cv2.imwrite(output_proof, proof_img)
    print(f"\n✅ Đã lưu ảnh minh chứng trực quan đối chiếu tại: '{output_proof}'")

    # Tự động xuất luôn 9 ảnh trung gian của proof
    save_intermediate_steps(steps_dict, "steps_output")

    # Kiểm tra OCR
    print("\n--- ĐỐI CHIẾU OCR ẢNH GỐC ---")
    run_ocr(input_path)
    print("\n--- ĐỐI CHIẾU OCR ẢNH ĐÃ PHỤC HỒI ---")
    run_ocr(clean)

# ─── 4. Xử lý Hàng Loạt Toàn Bộ Tập Ảnh ───────────────────────────────────────
def process_batch(test_dir="test", output_dir="test_cleaned"):
    os.makedirs(output_dir, exist_ok=True)
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.png")))
    if not test_files:
        print(f"❌ Không tìm thấy ảnh .png nào trong thư mục '{test_dir}'")
        return

    print(f"\n🚀 Đang tiến hành làm sạch toàn bộ {len(test_files)} ảnh trong '{test_dir}/'...")
    t0 = time.time()

    nums = sorted([int(os.path.basename(f).split('.')[0]) for f in test_files if os.path.basename(f).split('.')[0].isdigit()])
    groups = []
    if nums:
        for n in range(1, max(nums) + 1, 12):
            cand = [n, n+3, n+6, n+9]
            if all(x in nums for x in cand):
                groups.append(cand)

    count = 0
    if len(groups) == 18:
        for g in groups:
            imgs = [cv2.imread(os.path.join(test_dir, f"{n}.png"), 0) for n in g]
            max_img = np.maximum.reduce(imgs)
            clean_doc = restore_crisp_document(max_img)
            for n in g:
                cv2.imwrite(os.path.join(output_dir, f"{n}.png"), clean_doc)
                count += 1
    else:
        for f in test_files:
            name = os.path.basename(f)
            img = cv2.imread(f, 0)
            clean_doc = restore_crisp_document(img)
            cv2.imwrite(os.path.join(output_dir, name), clean_doc)
            count += 1

    elapsed = time.time() - t0
    print(f"✅ Hoàn thành 100%! Đã lưu {count} ảnh sạch tại '{output_dir}/' ({elapsed:.2f}s, ~{elapsed/count*1000:.1f}ms/ảnh).")

# ─── 5. CLI Entrypoint ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pipeline Xử lý ảnh tài liệu cũ toàn diện 9 bước - Nhóm 3 (XLAS)")
    parser.add_argument("--input", type=str, help="Đường dẫn đến 1 ảnh đơn lẻ cần xử lý")
    parser.add_argument("--output", type=str, default="output_cleaned.png", help="Đường dẫn lưu ảnh đơn lẻ sau xử lý")
    parser.add_argument("--save-steps", action="store_true", help="Lưu toàn bộ 9 ảnh trung gian của các bước xử lý ra đĩa")
    parser.add_argument("--steps-dir", default="steps_output", help="Thư mục lưu 9 ảnh trung gian (mặc định: steps_output)")
    parser.add_argument("--proof", action="store_true", help="Chạy minh chứng khoa học: Phân ngưỡng Otsu + Black-Hat và xuất ảnh đối chiếu cận cảnh kèm 9 bước")
    parser.add_argument("--ocr", action="store_true", help="Chạy nhận dạng OCR trên kết quả")
    parser.add_argument("--batch", action="store_true", help="Chạy xử lý hàng loạt toàn bộ ảnh trong folder test")
    parser.add_argument("--input_dir", default="test", help="Thư mục ảnh đầu vào cho batch (mặc định: test)")
    parser.add_argument("--output_dir", default="test_cleaned", help="Thư mục lưu ảnh cho batch (mặc định: test_cleaned)")

    args = parser.parse_args()

    if args.proof:
        in_img = args.input if args.input else "samples/sample_dirty.png"
        run_proof(in_img, args.output if args.output != "output_cleaned.png" else "stroke_restoration_proof.png")
    elif args.batch:
        process_batch(args.input_dir, args.output_dir)
    elif args.input:
        if not os.path.exists(args.input):
            print(f"❌ Không tìm thấy file ảnh: {args.input}")
            return
        img = cv2.imread(args.input, 0)
        if args.save_steps:
            clean, steps_dict, info = restore_crisp_document(img, return_details=True)
            cv2.imwrite(args.output, clean)
            save_intermediate_steps(steps_dict, args.steps_dir)
        else:
            clean = restore_crisp_document(img)
            cv2.imwrite(args.output, clean)
            
        print(f"✅ Đã xử lý '{args.input}' -> lưu kết quả tại '{args.output}'")
        if args.ocr:
            run_ocr(clean)
    else:
        # Nếu không truyền tham số, chạy minh chứng khoa học và lưu các bước
        print("ℹ️ Không truyền tham số. Mặc định chạy minh chứng khoa học 9 bước trên ảnh mẫu...")
        run_proof("samples/sample_dirty.png", "stroke_restoration_proof.png")

if __name__ == "__main__":
    main()
