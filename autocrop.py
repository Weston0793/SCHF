import torch
import numpy as np
from PIL import Image

def autocrop_image(image_tensor, cropmodel, device, threshold=0.5):
    cropmodel.eval()
    with torch.no_grad():
        output_mask = cropmodel(image_tensor.to(device)).cpu().numpy()

    output_mask_2d = output_mask[0, 0]
    thresholded_mask = (output_mask_2d > threshold).astype(np.uint8)

    y, x = np.where(thresholded_mask)
    if len(x) == 0 or len(y) == 0:
        return None, None

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    roi = image_tensor[0, 0].cpu().numpy()[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None, None

    cropped_tensor = torch.tensor(roi).unsqueeze(0).unsqueeze(0)  # Convert to a tensor with appropriate dimensions
    return cropped_tensor, (x_min, y_min, x_max, y_max)


