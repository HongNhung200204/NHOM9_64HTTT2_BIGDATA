import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def preprocess_image(img_path, size=(512, 512)):
    """
    GIAI ĐOẠN SPLITTING: Đọc ảnh và chuẩn hóa kích thước.
    """
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil.resize(size))
    return img_np

def create_gray_canvas(shape, color_value=128):
    """
    Tạo phông nền xám trung tính (Neutral Gray) theo chuẩn bài báo CObL.
    """
    return np.full(shape, color_value, dtype=np.uint8)

def extract_object_layer(img_np, mask, bg_color=128):
    """
    GIAI ĐOẠN MAP: Trích xuất vật thể từ ảnh gốc và đặt lên nền xám.
    """
    # Đảm bảo mask là 3 chiều để nhân ma trận
    if len(mask.shape) == 2:
        mask_3d = mask[:, :, np.newaxis]
    else:
        mask_3d = mask
        
    gray_bg = np.full_like(img_np, bg_color)
    # Công thức: (Ảnh gốc * Mặt nạ) + (Nền xám * Mặt nạ nghịch đảo)
    obj_layer = (img_np * mask_3d + gray_bg * (1 - mask_3d)).astype(np.uint8)
    return obj_layer

def alpha_composite_recursive(layers, masks):
    """
    GIAI ĐOẠN REDUCE: Ghép các lớp theo công thức đệ quy (Eq. 1 trong bài báo).
    x_i = (x_i * m_i + m_prev * x_prev * (1 - m_i)) / (m_total + delta)
    """
    composite = layers[0].copy() # Bắt đầu từ Background
    
    for i in range(1, len(layers)):
        mask = masks[i][:, :, np.newaxis].astype(np.float32)
        # Chồng lớp i lên lớp i-1
        composite = (layers[i] * mask + composite * (1 - mask)).astype(np.uint8)
        
    return composite

def save_batch_result(original, bg, objects, save_path):
    """
    GIAI ĐOẠN OUTPUT: Tạo ảnh dàn hàng ngang chuyên nghiệp để báo cáo.
    """
    num_plots = 2 + len(objects)
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 5))
    
    # Hiển thị ảnh gốc
    axes[0].imshow(original)
    axes[0].set_title("Input Image", fontweight='bold')
    
    # Hiển thị nền
    axes[1].imshow(bg)
    axes[1].set_title("Background", color='red')
    
    # Hiển thị từng lớp vật thể
    for i, obj in enumerate(objects):
        axes[i+2].imshow(obj["img"])
        axes[i+2].set_title(f"Layer {i+1}: {obj['name'].capitalize()}")
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close() # Đóng figure để tiết kiệm bộ nhớ cho Big Data