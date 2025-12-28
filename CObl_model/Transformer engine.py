import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

class TransformerEngine:
    """
    Đây là TRÁI TIM của hệ thống (AI Brain).
    Thực hiện: Semantic Encoding & Feature Extraction.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Nạp mô hình Transformer-based (Mask R-CNN)
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn(weights=weights).to(self.device).eval()
        self.transform = weights.transforms()
        self.categories = weights.meta["categories"]

    def process_image(self, img_pil):
        """
        Giai đoạn xử lý của Transformer: Biến ảnh thành các Layer Latent
        """
        img_t = self.transform(img_pil).to(self.device)
        with torch.no_grad():
            preds = self.model([img_t])[0]
        return preds