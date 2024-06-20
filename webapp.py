import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
from torchvision.models import MobileNet_V2_Weights, MobileNet_V3_Small_Weights
import numpy as np
import cv2
from skimage import exposure
import matplotlib.pyplot as plt


# Function to load model weights
def load_custom_model_weights(model, model_weights_path):
    checkpoint = torch.load(model_weights_path, map_location='cpu')
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)
    return model

# Define your models
class BinaryMobileNetV2(nn.Module):
    def __init__(self):
        super(BinaryMobileNetV2, self).__init__()
        base_model = models.mobilenet_v2()
        base_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        base_model.classifier[-1] = nn.Linear(in_features=1280, out_features=1)
        self.base_model = base_model

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.classifier(x)
        return torch.sigmoid(x)

class BinaryMobileNetV3Small(nn.Module):
    def __init__(self):
        super(BinaryMobileNetV3Small, self).__init__()
        base_model = models.mobilenet_v3_small()
        base_model.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        base_model.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=1, bias=True),
            nn.Sigmoid()
        )
        self.base_model = base_model

    def forward(self, x):
        x = self.base_model.features(x)
        x = x.mean([2, 3])
        x = self.base_model.classifier(x)
        return x

# Initialize models
lion_model = BinaryMobileNetV3Small()
swdsgd_model = BinaryMobileNetV2()

# Load the twin models
models_folder = "models"
lion_model = load_custom_model_weights(lion_model, f"{models_folder}/LionMobileNetV3Small.pth")
swdsgd_model = load_custom_model_weights(swdsgd_model, f"{models_folder}/SWDSGDMobileNetV2.pth")

# Define image enhancement functions
def adaptive_histogram_equalization(image):
    img = Image.fromarray(image)
    img = img.convert('L')
    equalized_image = ImageOps.equalize(img)
    return np.array(equalized_image)

def sharpen_image(image, alpha=1.5, beta=-0.5):
    blurred = image.filter(ImageFilter.GaussianBlur(1))
    sharpened = Image.blend(image, blurred, alpha)
    return np.array(sharpened)

def contrast_stretching(image):
    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    return img_rescale

# Define CAM functionality
def generate_cam(model, inputs):
    model.eval()
    feature_maps = None
    def hook_fn(module, input, output):
        nonlocal feature_maps
        feature_maps = output
    
    # Register hook to the last convolutional layer
    hook = model.base_model.features[-1].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        outputs = model(inputs)
    
    hook.remove()
    
    weights = model.base_model.classifier[0].weight.detach().cpu().numpy()
    cam = np.zeros(feature_maps.shape[2:], dtype=np.float32)
    
    for i, w in enumerate(weights[0]):
        cam += w * feature_maps[0, i].detach().cpu().numpy()
    
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (inputs.shape[2], inputs.shape[3]))
    colorized_cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return np.float32(colorized_cam) / 255

# Streamlit app layout
st.title("Pediatric Supracondylar Humerus Fracture X-Ray Classification with Twin Network")
uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Read the image using PIL
        image = Image.open(uploaded_file).convert('L')
        image_np = np.array(image)

        st.image(image_np, caption='Uploaded X-Ray', use_column_width=True)

        # Apply image enhancements
        enhanced_image = adaptive_histogram_equalization(image_np)
        enhanced_image = Image.fromarray(enhanced_image)
        enhanced_image = sharpen_image(enhanced_image)
        enhanced_image = contrast_stretching(np.array(enhanced_image))

        st.image(enhanced_image, caption='Enhanced X-Ray', use_column_width=True)

        # Resize for prediction
        enhanced_image_resized = Image.fromarray(enhanced_image).resize((200, 200))

        # Prepare the enhanced image for prediction
        enhanced_image_tensor = transforms.ToTensor()(enhanced_image_resized).unsqueeze(0)

        # Perform predictions
        with torch.no_grad():
            lion_output = lion_model(enhanced_image_tensor)
            swdsgd_output = swdsgd_model(enhanced_image_tensor)

        # Decision logic for twin network
        if lion_output > 0.5:
            prediction = "Fractured Pediatric Supracondylar Humerus"
            confidence = lion_output.item()
            cam = generate_cam(lion_model, enhanced_image_tensor)
        else:
            prediction = "Normal"
            confidence = 1 - swdsgd_output.item()
            cam = generate_cam(swdsgd_model, enhanced_image_tensor)

        st.write(f"Prediction: {prediction}")
        st.write(f"Confidence: {confidence:.2f}")

        # Display CAM
        original_image = enhanced_image_tensor.squeeze().cpu().numpy() * 255
        original_image = cv2.resize(original_image, (200, 200))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB) / 255
        overlay = np.uint8(255 * (0.5 * cam + 0.5 * original_image))

        st.image(original_image, caption='Original Image', use_column_width=True)
        st.image(overlay, caption='CAM Overlay', use_column_width=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")
