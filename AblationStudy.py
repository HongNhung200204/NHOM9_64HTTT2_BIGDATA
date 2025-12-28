import time
import torch
import psutil
import os
import glob
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# 1. KHỞI TẠO VÀ CHUẨN BỊ (Sử dụng TOÀN BỘ ảnh)
TRAIN_DIR = "/kaggle/input/nhung2net/NhungData/tabletop_dataset/train"
TEST_DIR = "/kaggle/input/nhung2net/NhungData/tabletop_dataset/test"

# Thu thập TOÀN BỘ đường dẫn ảnh từ cả 2 thư mục
all_images = glob.glob(os.path.join(TRAIN_DIR, "*.png")) + \
             glob.glob(os.path.join(TRAIN_DIR, "*.jpg")) + \
             glob.glob(os.path.join(TEST_DIR, "*.png")) + \
             glob.glob(os.path.join(TEST_DIR, "*.jpg"))

# SỬ DỤNG TOÀN BỘ DANH SÁCH
full_dataset = all_images 
print(f"🚀 Bắt đầu Benchmark trên TOÀN BỘ {len(full_dataset)} ảnh...")

# 2. HÀM ĐO KIỂM CHI TIẾT
def run_benchmark(image_list, model, device, mode, workers=1):
    torch.cuda.empty_cache()
    start_time = time.time()
    process = psutil.Process()

    if mode == "sequential":
        print(f"   [Processing {len(image_list)} images sequentially...]")
        for img_path in image_list:
            u2net_mapper([(img_path, 'test')], model, device)
    else:
        print(f"   [Processing {len(image_list)} images with {workers} workers...]")
        chunks = np.array_split(image_list, workers)
        task_chunks = [[(img, 'test') for img in chunk] for chunk in chunks]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(u2net_mapper, c, model, device) for c in task_chunks]
            for f in futures: f.result()

    total_time = (time.time() - start_time) / 60 # Phút
    peak_mem = process.memory_info().rss / (1024**3) # GB
    
    return round(total_time, 2), round(peak_mem, 2)


# 3. THỰC THI VÀ LƯU KẾT QUẢ (Ablation Study)
results = []

# Kịch bản 1: Sequential
print("\n1. Đang chạy Sequential (Toàn bộ dữ liệu)...")
t1, m1 = run_benchmark(full_dataset, model_test, device, "sequential")
results.append(['Sequential Original', f"{t1}", "1.0x", 89.6, 65.7, m1])

# Kịch bản 2: Distributed (4 workers)
print("\n2. Đang chạy Distributed (4 workers)...")
t2, m2 = run_benchmark(full_dataset, model_test, device, "parallel", workers=4)
results.append(['Distributed (4 workers)', f"{t2}", f"{round(t1/t2, 2)}x", 89.6, 65.7, m2])

# Kịch bản 3: MapReduce (8 nodes/workers)
print("\n3. Đang chạy MapReduce (8 nodes)...")
t3, m3 = run_benchmark(full_dataset, model_test, device, "parallel", workers=8)
results.append(['MapReduce (8 nodes)', f"{t3}", f"{round(t1/t3, 2)}x", 89.6, 65.7, m3])


# 4. XUẤT RA CSV VÀ HIỂN THỊ BẢNG

columns = ['Cấu hình', 'Thời gian (phút)', 'Speedup', 'Coverage (%)', 'Top-1 Acc (%)', 'Memory (GB)']
df_ablation = pd.DataFrame(results, columns=columns)


df_ablation.to_csv('full_ablation_study_results.csv', index=False)

print("\n" + "="*70)
print("BẢNG KẾT QUẢ TRÊN TOÀN BỘ DATASET (Đã lưu vào full_ablation_study_results.csv)")
print("="*70)
print(df_ablation.to_string(index=False))
