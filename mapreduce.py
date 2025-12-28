import torch
import os
import glob
import numpy as np
from PIL import Image
from torchvision import transforms
from u2net_model.u2net import U2NETP 
from concurrent.futures import ThreadPoolExecutor

# 1. CẤU HÌNH ĐƯỜNG DẪN
MODEL_PATH = "/kaggle/input/m/omaica/nhungnet/pytorch/default/1/u2net_best_nhungdata.pth"
TRAIN_DIR = "/kaggle/input/nhung2net/NhungData/tabletop_dataset/train"
TEST_DIR = "/kaggle/input/nhung2net/NhungData/tabletop_dataset/test"
OUTPUT_BASE_DIR = "/kaggle/working/tabletop_results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tạo cấu trúc thư mục đầu ra
for sub in ['train', 'test']:
    os.makedirs(os.path.join(OUTPUT_BASE_DIR, sub), exist_ok=True)

def load_model(model_path, device):
    net = U2NETP(3, 1)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
        
    if list(checkpoint.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            new_state_dict[k[7:]] = v
        net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(checkpoint)
    net.to(device)
    net.eval()
    return net
# 3. MAPPER: XỬ LÝ VÀ TRẢ VỀ DỮ LIỆU TRUNG GIAN
def u2net_mapper(image_info_chunk, model, device):
    """
    image_info_chunk: List các tuple (đường_dẫn_gốc, loại_thư_mục)
    """
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    intermediate_results = []
    for img_path, folder_type in image_info_chunk:
        img_name = os.path.basename(img_path)
        img = Image.open(img_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            d1,_,_,_,_,_,_ = model(input_tensor)
            pred = d1[:, 0, :, :]
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            mask_data = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
            
            # Lưu kết quả trung gian dưới dạng Key-Value
            intermediate_results.append({
                'name': img_name,
                'folder': folder_type,
                'mask': mask_data
            })
    return intermediate_results
# 4. CHẠY QUY TRÌNH MAPREDUCE
# Bước 1: Thu thập ảnh và gắn nhãn thư mục (Input & Splitting)
image_tasks = []
for img in glob.glob(os.path.join(TRAIN_DIR, "*.[jp][pn]g")):
    image_tasks.append((img, 'train'))
for img in glob.glob(os.path.join(TEST_DIR, "*.[jp][pn]g")):
    image_tasks.append((img, 'test'))

print(f"Tổng số ảnh: {len(image_tasks)}")

model_test = load_model(MODEL_PATH, device)

# Chia chunk (Splitting)
num_workers = 4
chunks = np.array_split(image_tasks, num_workers)

# Bước 2: Mapping
print("Đang thực hiện Mapping (Inference)...")
all_intermediate_data = []
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(u2net_mapper, chunk, model_test, device) for chunk in chunks]
    for f in futures:
        all_intermediate_data.extend(f.result())

# Bước 3: Reducing (Lưu file theo đúng cấu trúc thư mục)
print("Đang thực hiện Reducing (Lưu file)...")
for item in all_intermediate_data:
    save_filename = item['name'].rsplit('.', 1)[0] + ".png"
    save_path = os.path.join(OUTPUT_BASE_DIR, item['folder'], save_filename)
    
    # Chuyển mảng numpy thành ảnh và lưu
    mask_img = Image.fromarray(item['mask'])
    mask_img.save(save_path)

print(f"--- THÀNH CÔNG ---")
print(f"Kết quả Train: {len(os.listdir(os.path.join(OUTPUT_BASE_DIR, 'train')))} ảnh")
print(f"Kết quả Test: {len(os.listdir(os.path.join(OUTPUT_BASE_DIR, 'test')))} ảnh")
