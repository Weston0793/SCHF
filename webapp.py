import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps, ImageFilter
from torchvision.models import MobileNet_V2_Weights, MobileNet_V3_Small_Weights
import numpy as np
import cv2
from skimage import exposure
import matplotlib.pyplot as plt
import random
from cam_helper import get_cam, apply_cam_on_image  # Import the CAM helper functions

# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image

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
    equalized_image = ImageOps.equalize(image)
    return equalized_image

def sharpen_image(image, alpha=1.5, beta=-0.5):
    blurred = image.filter(ImageFilter.GaussianBlur(1))
    sharpened = Image.blend(image, blurred, alpha)
    return sharpened

def contrast_stretching(image):
    image_np = np.array(image)
    p2, p98 = np.percentile(image_np, (2, 98))
    img_rescale = exposure.rescale_intensity(image_np, in_range=(p2, p98))
    return Image.fromarray(img_rescale)

# Define function to create dataset with transformations
def create_transformed_dataset(image, batch_size=20):
    transform = transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop(200),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ToTensor()
    ])
    dataset = [image for _ in range(batch_size)]
    dataset = CustomDataset(images=dataset, transform=transform)
    return dataset

# Streamlit app layout
st.title("Pediatric Supracondylar Humerus Fracture X-Ray Classification with Twin Network")
uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Read the image using PIL
        image = Image.open(uploaded_file).convert('L')
        st.image(image, caption='Uploaded X-Ray', use_column_width=True)

        # Apply image enhancements
        enhanced_image = adaptive_histogram_equalization(image)
        enhanced_image = sharpen_image(enhanced_image)
        enhanced_image = contrast_stretching(enhanced_image)

        st.image(enhanced_image, caption='Enhanced X-Ray', use_column_width=True)

        # Create dataset with transformations
        transformed_dataset = create_transformed_dataset(enhanced_image, batch_size=20)
        dataloader = DataLoader(transformed_dataset, batch_size=1)

        lion_model.eval()
        swdsgd_model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize prediction variables
        lion_predictions = []
        swdsgd_predictions = []

        for inputs in dataloader:
            inputs = inputs.to(device)
            with torch.no_grad():
                lion_output = lion_model(inputs)
                lion_pred = (lion_output > 0.5).view(-1).long()
                lion_predictions.extend(lion_pred.cpu().numpy())

                # Only run swdsgd_model if lion_model predicts "Normal"
                if lion_pred == 0:
                    swdsgd_output = swdsgd_model(inputs)
                    swdsgd_pred = (swdsgd_output > 0.5).view(-1).long()
                    swdsgd_predictions.extend(swdsgd_pred.cpu().numpy())

        # Determine final prediction
        if any(pred == 1 for pred in lion_predictions):
            prediction = "Fractured Pediatric Supracondylar Humerus"
            confidence = np.mean(lion_predictions)
        else:
            if any(pred == 1 for pred in swdsgd_predictions):
                prediction = "Fractured Pediatric Supracondylar Humerus"
                confidence = np.mean(swdsgd_predictions)
            else:
                prediction = "Normal"
                confidence = 1 - np.mean(swdsgd_predictions)
        
        # Apply confidence threshold logic
        if confidence < 0.10:
            prediction = "Normal"
            confidence = 1 - confidence

        st.write(f"**Prediction:** {prediction}")
        st.write(f"**Confidence:** {confidence:.2f}")

        # Get and display CAM
        cam = get_cam(lion_model, inputs, 'base_model.features.12')  # Update with the correct layer name
        cam_image = apply_cam_on_image(np.array(enhanced_image.convert('RGB')), cam)  # Convert to RGB for color map application
        st.image(cam_image, caption='Class Activation Map (CAM)', use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
