import streamlit as st
import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from models import BinaryMobileNetV2, BinaryMobileNetV3Small, ResNetUNet
from autocrop import autocrop_image
from cam_helper import get_cam, apply_cam_on_image
from torch.utils.data import DataLoader
from enhancements import adaptive_histogram_equalization, sharpen_image, contrast_stretching
from augmentations import create_transformed_dataset
from predictions import load_models, predict_fracture

# Initialize models
lion_model = BinaryMobileNetV3Small()
swdsgd_model = BinaryMobileNetV2()
crop_model = ResNetUNet()

# Load models
models_folder = "models"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lion_model, swdsgd_model, crop_model = load_models(lion_model, swdsgd_model, crop_model, models_folder, device)

# Streamlit app layout
st.title("Pediatric Supracondylar Humerus X-Ray Fracture Detector employing a Twin Convolutional Neural Network")

# Style for larger text and highlighted prediction box
st.markdown("""
    <style>
    .large-text {
        font-size: 1.25rem;
    }
    .prediction-box {
        border: 2px solid #000;
        padding: 20px;
        background-color: #f0f0f0;
        font-size: 1.5rem;
    }
    .superscript {
        vertical-align: super;
        font-size: smaller;
    }
    </style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Read the image using PIL
        image = Image.open(uploaded_file).convert('L')
        st.image(image, caption='Uploaded X-Ray', use_column_width=True)

        # Apply new image enhancements on the original image
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
        st.markdown(f"<div class='prediction-box'><strong>Prediction:</strong> {prediction}<br><strong>Confidence:</strong> {confidence:.2f}</div>", unsafe_allow_html=True)

        # Generate and display CAM on the cropped image
        if cropped_image_pil is not None:
            cropped_image_gray = cropped_image_pil.convert('L')  # Ensure the image is in grayscale format
            
            img_tensor = transforms.ToTensor()(cropped_image_gray).unsqueeze(0).to(device)
            
            cam = get_cam(lion_model if prediction == "Fractured Pediatric Supracondylar Humerus" else swdsgd_model, img_tensor, 'base_model.features')
            
            cam_image = apply_cam_on_image(np.array(cropped_image_gray.convert('RGB')), cam)  # Convert to RGB for overlay
            st.image(cam_image, caption='Class Activation Map (CAM) on Cropped Image', use_column_width=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Additional Information
st.markdown("""
    ### Authors:
    Aba Lőrincz<sup class='superscript'>1,2,3,*</sup>, András Kedves<sup class='superscript'>2</sup>, Hermann Nudelman<sup class='superscript'>1,3</sup>, András Garami<sup class='superscript'>1</sup>, Gergő Józsa<sup class='superscript'>1,3</sup>, and Zsolt Kisander<sup class='superscript'>2</sup>
    
    ### Affiliations:
    1. Department of Thermophysiology, Institute for Translational Medicine, Medical School, University of Pécs, 12 Szigeti Street, H7624 Pécs, Hungary; aba.lorincz@gmail.com (AL); 
    2. Department of Automation, Faculty of Engineering and Information Technology, University of Pécs, 2 Boszorkány Street, H7624 Pécs, Hungary; 
    3. Division of Surgery, Traumatology, Urology, and Otorhinolaryngology, Department of Paediatrics, Clinical Complex, University of Pécs, 7 József Attila Street, H7623 Pécs, Hungary; 

    ### Code:
    [GitHub Repository](https://github.com/Weston0793/SCHF/)
""", unsafe_allow_html=True)
