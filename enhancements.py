import cv2
import numpy as np
from PIL import Image
from skimage import exposure

def adaptive_histogram_equalization(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    equalized_image = clahe.apply(np.array(image))
    return Image.fromarray(equalized_image)

def sharpen_image(image, sigma=1, alpha=1.5, beta=-0.5):
    image_np = np.array(image)
    blurred = cv2.GaussianBlur(image_np, (0, 0), sigma)
    sharpened = cv2.addWeighted(image_np, alpha, blurred, beta, 0)
    return Image.fromarray(sharpened)

def contrast_stretching(image):
    image_np = np.array(image)
    p2, p98 = np.percentile(image_np, (2, 98))
    img_rescale = exposure.rescale_intensity(image_np, in_range=(p2, p98))
    return Image.fromarray(img_rescale)
