import os
import time
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
import argparse

# Import các module tự viết
from CObl_model.decomposer import CObLDecomposer
from cobl import CObLCompositor

# Cấu hình mặc định
INPUT_DEFAULT = "/kaggle/input" 
OUTPUT_DEFAULT = "results"

def mapper_task(img_path):
    """Worker function cho MapReduce"""
    try:
        # 1. Khởi tạo Model (Mỗi worker tự khởi tạo để tránh lỗi CUDA)
        decomposer = CObLDecomposer(device='cpu')
        compositor = CObLCompositor()
        
        # 2. Decompose (Tách lớp)
        l_x, l_m, _ = decomposer.decompose(img_path)
        if l_x is None or len(l_x) == 0:
            return None
        
        # 3. Recompose (Hợp nhất - Phần của bạn)
        final_img, _ = compositor.recursive_compositing(l_x, l_m)
        
        # 4. Lưu kết quả
        filename = os.path.basename(img_path)
        res_img = np.clip(final_img * 255, 0, 255).astype(np.uint8)
        res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
        
        # Trả về kết quả (ảnh đã xử lý + tên file)
        return (filename, res_img)
        
    except Exception as e:
        return f"Error: {e}"

def main():
    parser = argparse.ArgumentParser(description="CObL Inference System")
    parser.add_argument('--input', type=str, default=INPUT_DEFAULT, help="Input folder")
    parser.add_argument('--output', type=str, default=OUTPUT_DEFAULT, help="Output folder")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    # 1. Tìm ảnh
    all_images = []
    for root, _, files in os.walk(args.input):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                all_images.append(os.path.join(root, f))
    
    print(f"🚀 Found {len(all_images)} images. Starting MapReduce...")
    
    # 2. Map Phase
    start_time = time.time()
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(mapper_task, all_images)
        
    # 3. Reduce Phase (Lưu file)
    success = 0
    for res in results:
        if isinstance(res, tuple):
            fname, img_data = res
            save_path = os.path.join(args.output, fname)
            cv2.imwrite(save_path, img_data)
            success += 1
            print(f"✅ Saved: {fname}")
        else:
            if res: print(f"❌ {res}")
            
    print(f"\n🏆 Done! Processed {success}/{len(all_images)} images in {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    main()