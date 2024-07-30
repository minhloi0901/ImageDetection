# Image_detection_controller.py
import torch

class ImageDetectionConfig:
    MODEL_PATH = 'back_end/models/predict_model'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMAGE_SIZE = (512, 512)
    HF_TOKEN = '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'
    INPAINTING_DIFFUSER = 'runwayml/stable-diffusion-inpainting'
    FLOAT16 = True
    NUM_MASKS = 2
    BLUR_FACTOR = 10
    PATCH_SIZE = (64, 64)
    SEED = 0