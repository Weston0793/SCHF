import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from PIL import Image
from torchvision.models import MobileNet_V2_Weights, MobileNet_V3_Small_Weights
import numpy as np
import cv2
from skimage import exposure
import matplotlib.pyplot as plt

# Function to load model weights
def load_model_weights(model, model_weights_path):
    checkpoint = torch.load(model_weights_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model

# Define your models
class BinaryMobileNetV2(nn.Module):
    def __init__(self):
        super(BinaryMobileNetV2, self).__init__()
        base_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        base_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        base_model.classifier[-1] = nn.Linear(in_features=1280, out_features=1)
        self.base_model = base_model

    def forward(self, x):
        return torch.sigmoid(self.base_model(x))

class BinaryMobileNetV3Small(nn.Module):
    def __init__(self):
        super(BinaryMobileNetV3Small, self).__init__()
        base_model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        base_model.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        base_model.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=1, bias=True),
            nn.Sigmoid()
        )
        self.base_model = base_model

    def forward(self, x):
        return self.base_model(x)

# Initialize models
lion_model = BinaryMobileNetV3Small()
swdsgd_model = BinaryMobileNetV2()

# Load the segmentation model
class ResNetUNet(nn.Module):
    def __init__(self):
        super(ResNetUNet, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        )
        self.middle = resnet.layer3
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return torch.sigmoid(x3)

cropmodel = ResNetUNet()
cropmodel.load_state_dict(torch.load('models/best_model_cropper.pth', map_location=torch.device('cpu')))
cropmodel.eval()

# Define image enhancement functions
def adaptive_histogram_equalization(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    equalized_image = clahe.apply(image)
    return equalized_image

def sharpen_image(image, sigma=1, alpha=1.5, beta=-0.5):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, alpha, blurred, beta, 0)
    return sharpened

def contrast_stretching(image):
    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    return img_rescale

# Define CAM functionality
def generate_cam(model, inputs):
    model.eval()
    outputs = model(inputs)
    feature_maps = model.base_model.features[-1]  # Get the feature maps from the last layer of features
    weights = model.base_model.classifier[0].weight.detach().cpu().numpy()  # Adjust according to your model architecture
    cam = np.zeros(feature_maps.shape[2:])
    for k, w in enumerate(weights[0]):
        cam += w * feature_maps[0, k].detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam = cv2.resize(cam, (200, 200))
    colorized_cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return np.float32(colorized_cam) / 255

# Load the twin models
models_folder = "models"
lion_model = load_model_weights(lion_model, f"{models_folder}/LionMobileNetV3Small.pth")
swdsgd_model = load_model_weights(swdsgd_model, f"{models_folder}/SWDSGDMobileNetV2.pth")

# Streamlit app layout
st.title("Pediatric Supracondylar Humerus Fracture X-Ray Classification with Twin Network")
uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Read the image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

        st.image(image, caption='Uploaded X-Ray', use_column_width=True)

        # Preprocess the image for segmentation
        image_resized = cv2.resize(image, (256, 256))
        image_tensor = torch.tensor(image_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Get segmentation mask
        with torch.no_grad():
            output_mask = cropmodel(image_tensor).squeeze().numpy()

        # Threshold the mask to create binary mask
        thresholded_mask = (output_mask > 0.5).astype(np.uint8)

        # Find bounding box coordinates for cropping
        y, x = np.where(thresholded_mask)
        if len(x) == 0 or len(y) == 0:
            st.write("No region of interest found in the image.")
        else:
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)

            # Crop the ROI from the original image
            cropped_image = image[y_min:y_max, x_min:x_max]
            cropped_image = cv2.resize(cropped_image, (256, 256))

            # Apply image enhancements
            enhanced_image = adaptive_histogram_equalization(cropped_image)
            enhanced_image = sharpen_image(enhanced_image)
            enhanced_image = contrast_stretching(enhanced_image)

            st.image(enhanced_image, caption='Enhanced and Cropped X-Ray', use_column_width=True)

            # Resize for prediction
            enhanced_image_resized = cv2.resize(enhanced_image, (200, 200))

            # Prepare the enhanced image for prediction
            enhanced_image_tensor = torch.tensor(enhanced_image_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

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