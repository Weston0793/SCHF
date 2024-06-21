import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import BinaryMobileNetV2, BinaryMobileNetV3Small, ResNetUNet, load_standard_model_weights, load_direct_model_weights
from PIL import ImageOps, ImageEnhance, Image, ImageFilter
import random
from scipy.ndimage import gaussian_filter
import cv2

# Define custom augmentations
class RandomElasticDeform:
    def __init__(self, sigma=15, alpha=15):
        self.sigma = sigma
        self.alpha = alpha

    def __call__(self, img):
        img = ImageOps.grayscale(img)
        if random.random() < 0.5:
            shape = img.size[::-1]
            dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
            dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            x_shifted = (x + dx).astype(np.float32)
            y_shifted = (y + dy).astype(np.float32)
            img_np = np.array(img)
            img_np = cv2.remap(img_np, x_shifted, y_shifted, interpolation=cv2.INTER_CUBIC)
            img = Image.fromarray(img_np)
        return img

def random_color():
    return random.choice([(0), (128), (255)])

class RandomPadding:
    def __init__(self, min_pad=15, max_pad=45):
        self.min_pad = min_pad
        self.max_pad = max_pad

    def __call__(self, img):
        pad_left = random.randint(self.min_pad, self.max_pad)
        pad_top = random.randint(self.min_pad, self.max_pad)
        pad_right = random.randint(self.min_pad, self.max_pad)
        pad_bottom = random.randint(self.min_pad, self.max_pad)
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        pad_color = random_color()
        img = ImageOps.expand(img, padding, pad_color)
        return img

class RandomContrastStretch:
    def __init__(self, low=0.8, high=1.2):
        self.low = low
        self.high = high

    def __call__(self, img):
        factor = random.uniform(self.low, self.high)
        img = ImageEnhance.Contrast(img).enhance(factor)
        return img

class RandomGaussianBlur:
    def __init__(self, kernel_size=(3,3), sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        if random.random() < 0.5:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            img = img.filter(ImageFilter.GaussianBlur(sigma))
        return img

class RandomNoise:
    def __init__(self, mean=0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if random.random() < 0.5:
            noise = np.random.normal(self.mean, self.std, img.size)
            img = Image.fromarray(np.clip(np.array(img) + noise, 0, 255).astype(np.uint8))
        return img

class RandomSpeckleNoise:
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, img):
        if random.random() < 0.5:
            noise = np.random.normal(0, self.std, img.size)
            img = Image.fromarray(np.clip(np.array(img) + np.array(img) * noise, 0, 255).astype(np.uint8))
        return img

class RandomZoom:
    def __init__(self, zoom_range=(0.95, 1.2)):
        self.zoom_range = zoom_range

    def __call__(self, img):
        width, height = img.size
        zoom_factor = random.uniform(*self.zoom_range)
        new_width, new_height = int(width * zoom_factor), int(height * zoom_factor)
        img = img.resize((new_width, new_height), resample=Image.BICUBIC)
        if zoom_factor > 1:
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            img = img.crop((left, top, left + width, top + height))
        else:
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            img = ImageOps.expand(img, (left, top, left, top), fill=0)
        return img

class RandomInvert:
    def __call__(self, img):
        if random.random() < 0.5:
            img = ImageOps.invert(img)
        return img

# Define data transformations
data_transforms = transforms.Compose([
    RandomPadding(),
    RandomZoom(),
    transforms.Resize([240, 240]),
    transforms.RandomAffine(degrees=(-90, 90), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    RandomElasticDeform(),
    RandomContrastStretch(),
    RandomGaussianBlur(),
    RandomNoise(),
    RandomSpeckleNoise(),
    transforms.Grayscale(),
    RandomInvert(),
    transforms.ToTensor(),
])

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

def create_transformed_dataset(image, batch_size=20, augment=True):
    transform_list = [
        transforms.Resize([240, 240]),
        transforms.CenterCrop([200, 200])
    ]
    
    if augment:
        transform_list.extend([
            transforms.RandomAffine(degrees=(-90, 90), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomElasticDeform(),
            RandomContrastStretch(),
            RandomGaussianBlur(),
            RandomNoise(),
            RandomSpeckleNoise(),
            transforms.Grayscale(),
            RandomInvert()
        ])
    
    transform = transforms.Compose(transform_list)
    transformed_images = [transform(image) for _ in range(batch_size)]
    transformed_images = [transforms.ToTensor()(img) for img in transformed_images]
    return CustomDataset(images=transformed_images)

def load_models(lion_model, swdsgd_model, crop_model, models_folder, device):
    lion_model = load_standard_model_weights(lion_model, f"{models_folder}/LionMobileNetV3Small.pth", map_location='cpu')
    swdsgd_model = load_standard_model_weights(swdsgd_model, f"{models_folder}/SWDSGDMobileNetV2.pth", map_location='cpu')
    crop_model = load_direct_model_weights(crop_model, f"{models_folder}/best_model_cropper.pth", map_location='cpu')

    lion_model.to(device)
    swdsgd_model.to(device)
    crop_model.to(device)

    return lion_model, swdsgd_model, crop_model

def make_predictions(model, dataloader, device):
    model.eval()
    predictions = []

    for inputs in dataloader:
        inputs = inputs.to(device)
        with torch.no_grad():
            output = model(inputs)
            pred = (output > 0.5).view(-1).long()
            predictions.extend(pred.cpu().numpy())

    return predictions

def predict_fracture(lion_model, swdsgd_model, dataloader_lion, dataloader_swdsgd, device):
    lion_predictions = make_predictions(lion_model, dataloader_lion, device)
    lion_fractures = sum(lion_predictions)

    if lion_fractures >= 2:
        return "Fractured Pediatric Supracondylar Humerus", np.mean(lion_predictions)
    else:
        swdsgd_predictions = make_predictions(swdsgd_model, dataloader_swdsgd, device)
        swdsgd_fractures = sum(swdsgd_predictions)

        if swdsgd_fractures >= 2:
            return "Fractured Pediatric Supracondylar Humerus", np.mean(swdsgd_predictions)
        else:
            return "Normal", 1 - np.mean(swdsgd_predictions)
