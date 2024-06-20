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
        x = x.mean([2, 3])  # Global average pooling
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
        x = x.mean([2, 3])  # Global average pooling
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

        st.image(enhanced_image, caption='Enhanced X-Ray', use_column_width=True)

        # Resize for prediction
        enhanced_image_resized = Image.fromarray(enhanced_image).resize((200, 200))

        # Prepare the enhanced image for prediction
        enhanced_image_tensor = transforms.ToTensor()(enhanced_image_resized).unsqueeze(0)

        # Perform predictions
        with torch.no_grad():
            lion_output = lion_model(enhanced_image_tensor)

        if lion_output > 0.5:
            prediction = "Fractured Pediatric Supracondylar Humerus"
            confidence = lion_output.item()
        else:
            with torch.no_grad():
                swdsgd_output = swdsgd_model(enhanced_image_tensor)
            
            if swdsgd_output > 0.5:
                prediction = "Fractured Pediatric Supracondylar Humerus"
                confidence = swdsgd_output.item()
            else:
                prediction = "Normal Humerus"
                confidence = 1 - swdsgd_output.item()

        st.write(f"**Prediction:** {prediction}")
        st.write(f"**Confidence:** {confidence:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
