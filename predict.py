import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np
from argparse import ArgumentParser

def create_args():
    parser = ArgumentParser()
    parser.add_argument("--image-path", type=str, required=True, help="Path to the image to be predicted")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu", help="Device to run the model on")
    return parser.parse_args()

def load_model(model_path, device):
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return processor, model
def predict(image_path, processor, model, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class_prob = probabilities[0, predicted_class_idx].item()
        
    return predicted_class_idx, predicted_class_prob

def main(args):
    device = torch.device(args.device)
    processor, model = load_model(args.model_path, device)
    
    predicted_class_idx, predicted_class_prob = predict(args.image_path, processor, model, device)
    if predicted_class_idx == 0:
        prediction = "Real"
    else:
        prediction = "Fake"
        
    print(f"Prediction: {prediction}")
    print(f"Prediction confidence: {predicted_class_prob:.4f}")

if __name__ == "__main__":
    args = create_args()
    main(args)
