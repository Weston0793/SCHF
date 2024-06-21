import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights

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

class ResNetUNet(nn.Module):
    """
    ResNetUNet model for image segmentation.
    """

    def __init__(self):
        super(ResNetUNet, self).__init__()
        resnet = models.resnet18(weights=None)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2)
        self.middle = resnet.layer3
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return torch.sigmoid(x3)


def load_model_weights(model, checkpoint_path, model_type='standard'):
    """Load model weights from a checkpoint, handling different formats."""
    checkpoint = torch.load(checkpoint_path)
    print("Checkpoint keys:", checkpoint.keys())  # Print keys for debugging

    if model_type == 'standard':
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise KeyError(f"'model_state_dict' not found in the checkpoint file: {checkpoint_path}")
    elif model_type == 'direct':
        model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

# Example usage
crop_model = ResNetUNet()
crop_model = load_model_weights(crop_model, f"{models_folder}/best_model_cropper.pth", model_type='direct')


