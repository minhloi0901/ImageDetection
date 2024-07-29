import os
from argparse import ArgumentParser

import torch
from torchvision import transforms
from diffusers.pipelines.auto_pipeline import AutoPipelineForInpainting
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Define your function to compute MRE
def compute_MRE(
    pipeline,
    init_image: torch.Tensor,
    device: torch.device,
    num_masks: int,
    blur_factor: float,
    patch_size=(64, 64),
    seed: int = 0,
):
    C, W, H = init_image.size()
    image_size = (init_image.size(1), init_image.size(2))
    rng = torch.Generator(device).manual_seed(seed)

    patch_dims = (
        (image_size[0] + patch_size[0] - 1) // patch_size[0],
        (image_size[1] + patch_size[1] - 1) // patch_size[1],
    )
    ids_per_mask = (patch_dims[0] * patch_dims[1] + num_masks - 1) // num_masks

    alter_mask = torch.zeros((2, *image_size), device=device)
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            patch_x = i // patch_size[0]
            patch_y = j // patch_size[1]
            if (patch_x + patch_y) % 2:
                alter_mask[0, i, j] = 1
            else:
                alter_mask[1, i, j] = 1
    masks = alter_mask.unsqueeze(1).repeat((num_masks, 1, 1, 1)).to(device)

    blurred_masks = [None for _ in range(num_masks)]
    for k in range(num_masks):
        mask = transforms.ToPILImage()(masks[k].squeeze(0).cpu())
        blurred_masks[k] = transforms.ToTensor()(
            pipeline.mask_processor.blur(mask, blur_factor=blur_factor)
        ).to(device)

    image = init_image.clone()
    for mask in blurred_masks:
        tmp = pipeline(
            prompt="",
            image=image.unsqueeze(0).to(device),
            mask_image=mask.unsqueeze(0).to(device),
            generator=rng,
        ).images[0]
        image = transforms.ToTensor()(tmp).to(device)

    return torch.abs(image - init_image)

# Argument parser
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
    parser.add_argument("--seed", type=int, default=0, help="Seed used for Generator")

    return parser.parse_args()

def main(args):
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
    
    device = torch.device(args.device)
    args.device = device

    # Load and preprocess the image
    image = Image.open(args.image_path).convert('RGB')
    image = image.resize(args.image_size, Image.LANCZOS)
    image_tensor = transforms.ToTensor()(image).to(device)

    # Check the image tensor shape
    print(f"Preprocessed image shape: {image_tensor.shape}")

    # Load the diffusion model pipeline
    if args.float16:
        pipeline = AutoPipelineForInpainting.from_pretrained(
            args.diffuser, torch_dtype=torch.float16, variant="fp16"
        ).to(device)
    else:
        pipeline = AutoPipelineForInpainting.from_pretrained(args.diffuser).to(device)

    # Compute the MRE image
    mre_image_tensor = compute_MRE(
        pipeline=pipeline,
        init_image=image_tensor,
        device=device,
        num_masks=args.num_masks,
        blur_factor=args.blur_factor,
        patch_size=args.patch_size,
        seed=args.seed,
    )

    # Check the MRE image tensor shape
    print(f"MRE image tensor shape: {mre_image_tensor.shape}")

    # Load the trained ResNet model and processor
    processor = AutoImageProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageClassification.from_pretrained(args.model_path, trust_remote_code=True).to(device)

    # Prepare the MRE image for prediction
    mre_image_pil = transforms.ToPILImage()(mre_image_tensor.cpu())
    inputs = processor(images=mre_image_pil, return_tensors="pt").to(device)
    inputs = {key: val.squeeze(0) for key, val in inputs.items()}

    # Check the processed inputs shape
    print(f"Processed inputs shape: {inputs['pixel_values'].shape}")

    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

    print(f"Prediction: {'Real' if prediction == 0 else 'Fake'}")

if __name__ == "__main__":
    args = create_args()
    main(args)
