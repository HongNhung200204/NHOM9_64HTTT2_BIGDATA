import torch
import os
import time
import glob
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from u2net_model.u2net import U2NETP 
DATA_A = {
    "name": "TableTop Dataset",
    "img_dir": "/kaggle/input/nhung2net/NhungData/tabletop_dataset/test",
    "gt_dir": "/kaggle/working/tabletop_results/test"
}
DATA_B = {
    "name": "Data1",
    "img_dir": "/kaggle/input/nhung2net/NhungData/Data1/Image/test",
    "gt_dir": "/kaggle/input/nhung2net/NhungData/Data1/Mask/test"
}

MODEL_PATH = "/kaggle/input/m/omaica/nhungnet/pytorch/default/1/u2net_best_nhungdata.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. CÁC HÀM TÍNH CHỈ SỐ ĐÁNH GIÁ (METRICS)
def compute_mae(pred, gt):
    return np.mean(np.abs(pred - gt))

def compute_fmeasure(pred, gt):
    # Thresholding
    pred_bin = (pred > 0.5).astype(np.float32)
    gt_bin = (gt > 0.5).astype(np.float32)
    
    tp = np.sum(pred_bin * gt_bin)
    precision = tp / (np.sum(pred_bin) + 1e-8)
    recall = tp / (np.sum(gt_bin) + 1e-8)
    
    if (precision + recall) == 0: return 0
    f_score = (1.3 * precision * recall) / (0.3 * precision + recall) # beta^2 = 0.3
    return f_score


def evaluate_on_dataset(data_info, model, device):
    img_paths = glob.glob(os.path.join(data_info["img_dir"], "*.[jp][pn]g")) # Lấy 100 ảnh để test nhanh
    
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    mae_scores = []
    f_scores = []
    start_time = time.time()
    
    print(f"-> Đang đánh giá trên: {data_info['name']}")
    
    for img_path in img_paths:
        img_name = os.path.basename(img_path).split('.')[0]
        gt_path = os.path.join(data_info["gt_dir"], img_name + ".png")
        
        if not os.path.exists(gt_path): continue
            
        # Dự đoán
        img = Image.open(img_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            d1,_,_,_,_,_,_ = model(input_tensor)
            pred = d1[:, 0, :, :]
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            pred_np = pred.squeeze().cpu().numpy()
            
        # Ground Truth
        gt = Image.open(gt_path).convert('L').resize((320, 320))
        gt_np = np.array(gt) / 255.0
        
        # Tính metrics
        mae_scores.append(compute_mae(pred_np, gt_np))
        f_scores.append(compute_fmeasure(pred_np, gt_np))
        
    avg_time = (time.time() - start_time) / len(img_paths)
    
    return {
        "Tên bộ dữ liệu": data_info["name"],
        "Số lượng ảnh": len(img_paths),
        "MAE (↓)": round(np.mean(mae_scores), 4),
        "F-measure (↑)": round(np.mean(f_scores), 4),
        "Thời gian/Ảnh (s)": round(avg_time, 4)
    }
net = U2NETP(3, 1) 
checkpoint = torch.load(MODEL_PATH, map_location=device)
if 'state_dict' in checkpoint: checkpoint = checkpoint['state_dict']
net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
net.to(device).eval()

# Chạy đánh giá cho cả 2 bộ
res_a = evaluate_on_dataset(DATA_A, net, device)
res_b = evaluate_on_dataset(DATA_B, net, device) 


# Hiển thị kết quả
df_comparison = pd.DataFrame([res_a, res_b])
print("\n" + "="*60)
print("BẢNG SO SÁNH HIỆU QUẢ TRÊN 2 BỘ DỮ LIỆU")
print("="*60)
print(df_comparison.to_string(index=False))
