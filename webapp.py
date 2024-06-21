import streamlit as st
import torch
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter
from torch.utils.data import Dataset, DataLoader
from models import BinaryMobileNetV2, BinaryMobileNetV3Small, ResNetUNet, load_standard_model_weights, load_direct_model_weights
from autocrop import autocrop_image
from skimage import exposure
from cam_helper import get_cam, apply_cam_on_image
from predict import create_transformed_dataset, load_models, predict_fracture, CustomDataset

# Initialize models
lion_model = BinaryMobileNetV3Small()
swdsgd_model = BinaryMobileNetV2()
crop_model = ResNetUNet()

# Load models
models_folder = "models"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lion_model, swdsgd_model, crop_model = load_models(lion_model, swdsgd_model, crop_model, models_folder, device)

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

# Streamlit app layout
st.title("Pediatric Supracondylar Humerus Fracture X-Ray Classification with Twin Network")
uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Read the image using PIL
        image = Image.open(uploaded_file).convert('L')
        st.image(image, caption='Uploaded X-Ray', use_column_width=True)

        # Apply image enhancements on the original image
        enhanced_image = adaptive_histogram_equalization(image)
        enhanced_image = sharpen_image(enhanced_image)
        enhanced_image = contrast_stretching(enhanced_image)

        st.image(enhanced_image, caption='Enhanced X-Ray', use_column_width=True)

        # Autocrop the enhanced image
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        image_tensor = transform(enhanced_image).unsqueeze(0).to(device)
        cropped_image_pil, _ = autocrop_image(image_tensor, crop_model, device)
        
        if cropped_image_pil is not None:
            st.image(cropped_image_pil, caption='Cropped X-Ray', use_column_width=True)
        else:
            st.warning("No region of interest found. Using original image.")
            cropped_image_pil = enhanced_image

        # Create datasets with and without augmentation
        dataloader_lion = DataLoader(create_transformed_dataset(cropped_image_pil, batch_size=20, augment=True), batch_size=1)
        dataloader_swdsgd = DataLoader(create_transformed_dataset(cropped_image_pil, batch_size=20, augment=True), batch_size=1)
        dataloader_no_augment = DataLoader(create_transformed_dataset(cropped_image_pil, batch_size=1, augment=False), batch_size=1)

        # Get the prediction
        prediction, confidence = predict_fracture(lion_model, swdsgd_model, dataloader_lion, dataloader_swdsgd, device)

        # Display the prediction in a highlighted box
        st.markdown(f"<div style='border:2px solid #000; padding: 10px; background-color: #f0f0f0;'><strong>Prediction:</strong> {prediction}<br><strong>Confidence:</strong> {confidence:.2f}</div>", unsafe_allow_html=True)

        # Generate and display CAM on the cropped image
        if cropped_image_pil is not None:
            cropped_image_rgb = cropped_image_pil.convert('RGB')
            img_tensor = transforms.ToTensor()(cropped_image_rgb).unsqueeze(0).to(device)
            cam = get_cam(lion_model if prediction == "Fractured Pediatric Supracondylar Humerus" else swdsgd_model, img_tensor, 'base_model.features')
            cam_image = apply_cam_on_image(np.array(cropped_image_rgb), cam)
            st.image(cam_image, caption='Class Activation Map (CAM) on Cropped Image', use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
