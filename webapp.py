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

# Initialize models
lion_model = BinaryMobileNetV3Small()
swdsgd_model = BinaryMobileNetV2()
crop_model = ResNetUNet()

# Load the twin models and crop model
models_folder = "models"
lion_model = load_standard_model_weights(lion_model, f"{models_folder}/LionMobileNetV3Small.pth", map_location='cpu')
swdsgd_model = load_standard_model_weights(swdsgd_model, f"{models_folder}/SWDSGDMobileNetV2.pth", map_location='cpu')
crop_model = load_direct_model_weights(crop_model, f"{models_folder}/best_model_cropper.pth", map_location='cpu')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lion_model.to(device)
swdsgd_model.to(device)
crop_model.to(device)

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

        # Autocrop the image
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)
        cropped_image, crop_coords = autocrop_image(image_tensor, crop_model, device)
        
        if cropped_image is not None:
            st.image(cropped_image, caption='Cropped X-Ray', use_column_width=True)
            enhanced_image = cropped_image
        else:
            st.warning("No region of interest found. Using original image.")
            enhanced_image = image

        # Apply image enhancements
        enhanced_image = adaptive_histogram_equalization(enhanced_image)
        enhanced_image = sharpen_image(enhanced_image)
        enhanced_image = contrast_stretching(enhanced_image)

        st.image(enhanced_image, caption='Enhanced X-Ray', use_column_width=True)

        # Create dataset with transformations
        transformed_dataset = create_transformed_dataset(enhanced_image, batch_size=20)
        dataloader = DataLoader(transformed_dataset, batch_size=1)

        lion_model.eval()
        swdsgd_model.eval()

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

        # Generate and display CAM
        if cropped_image is not None and crop_coords is not None:
            x_min, y_min, x_max, y_max = crop_coords
            img_tensor = transforms.ToTensor()(enhanced_image).unsqueeze(0).to(device)
            cam = get_cam(lion_model if any(pred == 1 for pred in lion_predictions) else swdsgd_model, img_tensor, 'base_model.features')
            
            # Resize CAM to the cropped image size
            cam_resized = cv2.resize(cam, (x_max - x_min, y_max - y_min))
            original_image_np = np.array(image)

            # Create a full-size CAM mask with three channels
            full_size_cam = np.zeros((*original_image_np.shape, 3), dtype=np.uint8)
            cam_resized_rgb = np.repeat(cam_resized[:, :, np.newaxis], 3, axis=2)
            full_size_cam[y_min:y_max, x_min:x_max] = cam_resized_rgb
            
            cam_image = apply_cam_on_image(original_image_np, full_size_cam)
            st.image(cam_image, caption='Class Activation Map (CAM) on Original Image', use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
