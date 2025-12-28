import cv2
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

class MapReduceManager:
    """
    Đây là CỖ MÁY ĐIỀU PHỐI (Big Data Controller).
    Thực hiện: Splitting -> Map -> Reduce.
    """
    def __init__(self, transformer_engine, output_dir="/kaggle/working/output_batch"):
        self.engine = transformer_engine
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def stage_map(self, img_path, img_name):
        """
        Giai đoạn MAP: Xử lý song song từng ảnh và bóc tách lớp
        """
        img_pil = Image.open(img_path).convert("RGB").resize((512, 512))
        img_np = np.array(img_pil)
        
        # Gọi bộ não Transformer xử lý
        preds = self.engine.process_image(img_pil)
        
        gray_bg = np.full_like(img_np, 128) # Phông xám học thuật
        combined_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        
        # Mỗi vòng lặp này mô phỏng 1 Mapper GPU xử lý 1 Layer vật thể
        count = 0
        for i in range(min(len(preds['masks']), 5)):
            if preds['scores'][i] > 0.2:
                mask = (preds['masks'][i, 0].cpu().numpy() > 0.5).astype(np.uint8)
                label = self.engine.categories[preds['labels'][i]]
                
                # Trích xuất pixel vật thể (Decoder & Masking)
                obj_layer = (img_np * mask[:,:,None] + gray_bg * (1 - mask[:,:,None])).astype(np.uint8)
                
                # Lưu trữ tạm kết quả của Mapper
                self.save_layer(obj_layer, img_name, f"layer_{count}_{label}")
                
                # Cập nhật mask tổng cho giai đoạn Reduce
                combined_mask = np.maximum(combined_mask, mask)
                count += 1
        
        return img_np, combined_mask, gray_bg

    def stage_reduce(self, img_np, combined_mask, gray_bg, img_name):
        """
        Giai đoạn REDUCE: Tổng hợp các lớp để kiểm tra tính nhất quán (Background)
        """
        mask_inv = 1 - combined_mask[:, :, None]
        bg_layer = (img_np * mask_inv + gray_bg * (1 - mask_inv)).astype(np.uint8)
        self.save_layer(bg_layer, img_name, "background_final")

    def save_layer(self, img_array, img_name, layer_type):
        folder = os.path.join(self.output_dir, img_name.split('.')[0])
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{layer_type}.png")
        cv2.imwrite(path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

    def run_batch(self, input_dir, limit=10):
        files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])[:limit]
        print(f"🚀 [Quy trình 3] Bắt đầu xử lý lô dữ liệu {len(files)} ảnh...")
        
        for f in tqdm(files):
            path = os.path.join(input_dir, f)
            # Thực hiện MAP
            img_np, comb_mask, g_bg = self.stage_map(path, f)
            # Thực hiện REDUCE
            self.stage_reduce(img_np, comb_mask, g_bg, f)