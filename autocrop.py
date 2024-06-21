import torch
import numpy as np
from PIL import Image
import cv2

def autocrop_image(image_tensor, cropmodel, device, threshold=0.5):
    cropmodel.eval()
    with torch.no_grad():
        output_mask = cropmodel(image_tensor.unsqueeze(0).to(device)).cpu().numpy()

    output_mask_2d = output_mask[0, 0]
    thresholded_mask = (output_mask_2d > threshold).astype(np.uint8)

    y, x = np.where(thresholded_mask)
    if len(x) == 0 or len(y) == 0:
        return None

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    roi = image_tensor.numpy().squeeze()[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None

    return Image.fromarray((roi * 255).astype(np.uint8))
