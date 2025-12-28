
import torch
import numpy as np
import torch.nn as nn

class MidasDepthEstimator(nn.Module):
    def __init__(self, model_type="DPT_Large"):
        super().__init__()
        self.model_type = model_type
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def forward(self, img):
        device = next(self.midas.parameters()).device
        input_batch = self.transform(img).to(device)
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return prediction[None] / prediction.max()
