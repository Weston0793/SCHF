import torch
import numpy as np
from torch.utils.data import DataLoader
from models import load_standard_model_weights, load_direct_model_weights

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

    if lion_fractures >= 10:
        return "Fractured Pediatric Supracondylar Humerus", np.mean(lion_predictions)
    else:
        swdsgd_predictions = make_predictions(swdsgd_model, dataloader_swdsgd, device)
        swdsgd_fractures = sum(swdsgd_predictions)

        if swdsgd_fractures >= 10:
            return "Fractured Pediatric Supracondylar Humerus", np.mean(swdsgd_predictions)
        else:
            return "Normal", 1 - np.mean(swdsgd_predictions)
