import os
import torch
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from diffusers.pipelines.auto_pipeline import AutoPipelineForInpainting
from transformers import AutoImageProcessor, AutoModelForImageClassification
from huggingface_hub import login
import matplotlib.pyplot as plt

def create_args():
    parser = ArgumentParser()

    parser.add_argument("--image-path", type=str, required=True, help="Path to the image for prediction")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model for prediction")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu", help="Device to run the prediction on (cuda or cpu)")
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512), help="Size to resize the image (width height)")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging face access token")
    parser.add_argument("--diffuser", type=str, default="runwayml/stable-diffusion-inpainting", help="Diffuser to use")
    parser.add_argument("--float16", action="store_true")
    parser.add_argument("--num-masks", type=int, default=2, help="Number of masks")
    parser.add_argument("--blur-factor", type=float, default=10, help="Blur factor used for blurring masks")
    parser.add_argument("--patch-size", type=int, nargs=2, default=[64, 64], help="Size of patch used for computing MRE")

    return parser.parse_args()

def compute_MRE(pipeline, init_image, device, num_masks, blur_factor, patch_size, seed=0):
    C, W, H = init_image.size()
    image_size = (init_image.size(1), init_image.size(2))
    rng = torch.Generator(device).manual_seed(seed)

    patch_dims = (
        (image_size[0] + patch_size[0] - 1) // patch_size[0],
        (image_size[1] + patch_size[1] - 1) // patch_size[1],
    )
    alter_mask = torch.zeros((2, *image_size), device=device)
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            patch_x = i // patch_size[0]
            patch_y = j // patch_size[1]
            if (patch_x + patch_y) % 2:
                alter_mask[0, i, j] = 1
            else:
                alter_mask[1, i, j] = 1
    masks = alter_mask

    blurred_masks = [None for _ in range(num_masks)]
    for k in range(num_masks):
        mask = transforms.ToPILImage()(masks[k].cpu())  # Ensure mask is on CPU before converting
        blurred_masks[k] = transforms.ToTensor()(
            pipeline.mask_processor.blur(mask, blur_factor=blur_factor)
        ).to(device)

    image = init_image.clone()
    for mask in blurred_masks:
        tmp = pipeline(
            prompt="",
            image=image,
            mask_image=mask,
            generator=rng,
        ).images
        image = transforms.ToTensor()(tmp[0]).to(device)

    return torch.abs(image - init_image)

def predict(image_tensor, processor, model, device, image_size):
    inputs = processor(images=image_tensor, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    
    return predictions.item()

def main(args):
    if args.hf_token:
        login(token=args.hf_token)
    
    device = torch.device(args.device)
    
    # Load the image
    image = Image.open(args.image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(args.image_size),
    ])
    image_tensor = transform(image).to(device)
    
    # Load the pipeline for MRE computation directly from Hugging Face
    if args.float16:
        pipeline = AutoPipelineForInpainting.from_pretrained(args.diffuser, torch_dtype=torch.float16, variant="fp16").to(device)
    else:
        pipeline = AutoPipelineForInpainting.from_pretrained(args.diffuser).to(device)

    # Compute MRE
    mre_image_tensor = compute_MRE(
        pipeline=pipeline,
        init_image=image_tensor,
        device=device,
        num_masks=args.num_masks,
        blur_factor=args.blur_factor,
        patch_size=args.patch_size,
    ).squeeze(0)
    
    # Convert MRE tensor to PIL Image for display
    mre_image_pil = transforms.ToPILImage()(mre_image_tensor.cpu())

    # Display the MRE image using matplotlib
    plt.imshow(mre_image_pil)
    plt.title("MRE Image")
    plt.axis('off')
    plt.show()
    
    # Load the classification model and processor directly from Hugging Face
    processor = AutoImageProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    prediction = predict(mre_image_tensor, processor, model, device, args.image_size)

    if prediction == 0:
        print("The image is real.")
    else:
        print("The image is fake.")

if __name__ == "__main__":
    args = create_args()
    main(args)
#dfafdas
#aaaa