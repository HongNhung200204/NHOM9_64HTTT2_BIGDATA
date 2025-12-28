import cv2
import numpy as np
from ultralytics import YOLO

class CObLDecomposer:
    def __init__(self, model_name='yolov8n-seg.pt', device='cpu'):
        self.model = YOLO(model_name)
        self.device = device

    def decompose(self, img_path):
        """
        Tách ảnh thành các layers (Background sạch + Objects).
        """
        img = cv2.imread(img_path)
        if img is None: return None, None, None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Chạy model (Ép chạy CPU nếu cần multiprocessing)
        results = self.model(img_rgb, verbose=False, device=self.device)[0]
        
        layers_rgb = []
        layers_alpha = []
        
        if results.masks:
            masks = results.masks.data.cpu().numpy()
            masks = [cv2.resize(m, (img_rgb.shape[1], img_rgb.shape[0])) for m in masks]
            
            # 1. Tạo Clean Background (Amodal Completion giả lập)
            all_mask = np.sum(masks, axis=0) > 0.5
            all_mask_u8 = (all_mask * 255).astype(np.uint8)
            
            # Dilation để xóa sạch viền
            kernel = np.ones((15, 15), np.uint8)
            dilated = cv2.dilate(all_mask_u8, kernel, iterations=2)
            bg_clean = cv2.inpaint(img_rgb, dilated, 10, cv2.INPAINT_TELEA)
            
            layers_rgb.append(bg_clean.astype(np.float32) / 255.0)
            layers_alpha.append(np.ones_like(masks[0]))
            
            # 2. Tạo Object Layers (Sắp xếp xa -> gần)
            y_max = [np.max(np.where(m > 0.5)[0]) if np.any(m > 0.5) else 0 for m in masks]
            sorted_idx = np.argsort(y_max)
            
            for i in sorted_idx:
                layers_rgb.append(img_rgb.astype(np.float32) / 255.0)
                layers_alpha.append(masks[i])
                
        return layers_rgb, layers_alpha, img_rgb