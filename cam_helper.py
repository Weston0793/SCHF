import torch
import torch.nn.functional as F
import numpy as np
import cv2

def get_cam(model, img_tensor, target_layer):
    """
    Generate a Class Activation Map (CAM) for a given image and model.
    
    Args:
        model (nn.Module): The neural network model.
        img_tensor (torch.Tensor): The input image tensor.
        target_layer (str): The layer to target for CAM generation.
        
    Returns:
        np.ndarray: The CAM mask.
    """
    model.eval()
    
    def forward_hook(module, input, output):
        activation[0] = output
    
    activation = {}
    layer = dict([*model.named_modules()]).get(target_layer, None)
    if layer is None:
        raise ValueError(f"Layer {target_layer} not found in the model")
        
    hook = layer.register_forward_hook(forward_hook)
    
    with torch.no_grad():
        output = model(img_tensor)
    
    hook.remove()
    
    output = output[0]
    output = F.relu(output)
    weight_softmax_params = list(model.parameters())[-2].data.numpy()
    weight_softmax = np.squeeze(weight_softmax_params)
    
    activation = activation[0].squeeze().cpu().data.numpy()
    cam = np.zeros(activation.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weight_softmax):
        cam += w * activation[i, :, :]
    
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img_tensor.shape[-1], img_tensor.shape[-2]))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    
    return cam

def apply_cam_on_image(img, cam):
    """
    Apply the CAM mask on the image with an inverted colormap.
    
    Args:
        img (np.ndarray): The original image.
        cam (np.ndarray): The CAM mask.
        
    Returns:
        np.ndarray: The image with the CAM applied.
    """
    if len(cam.shape) == 2:  # If cam is single channel
        cam = np.repeat(cam[:, :, np.newaxis], 3, axis=2)
    
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))  # Ensure the CAM is resized to the image dimensions
    heatmap = cv2.applyColorMap(np.uint8(255 * (1 - cam)), cv2.COLORMAP_TWILIGHT_SHIFTED)  # Apply the CAM mask
    heatmap = np.float32(heatmap) / 255
    cam_img = heatmap + np.float32(img) / 255  # Normalize the image before adding
    cam_img = cam_img / np.max(cam_img)
    return np.uint8(255 * cam_img)
