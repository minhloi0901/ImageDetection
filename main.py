import os
from argparse import ArgumentParser

import torch
from torchvision import transforms
from diffusers.pipelines.auto_pipeline import AutoPipelineForInpainting
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F


def compute_MRE(
    pipeline,
    init_images: torch.Tensor,
    device: torch.device,
    num_masks: int,
    blur_factor: float,
    patch_size=(64, 64),
    seed: int = 0,
):
    N, C, W, H = init_images.size()
    image_size = (init_images.size(2), init_images.size(3))
    rng = torch.Generator(device).manual_seed(seed)

    patch_dims = (
        (image_size[0] + patch_size[0] - 1) // patch_size[0],
        (image_size[1] + patch_size[1] - 1) // patch_size[1],
    )
    ids_per_mask = (patch_dims[0] * patch_dims[1] + num_masks - 1) // num_masks
    s = set()
    # masks = [
    #     [torch.zeros(image_size, dtype=torch.uint8) for _ in range(N)]
    #     for _ in range(num_masks)
    # ]
    # for b in range(N):
    #     ids = torch.randperm(
    #         patch_dims[0] * patch_dims[1], generator=rng, device=device
    #     )
    #
    #     for ptr, id in enumerate(ids):
    #         k = ptr // ids_per_mask
    #
    #         patch_x = id // patch_dims[1]
    #         patch_y = id % patch_dims[1]
    #         for i in range(
    #             patch_x * patch_size[0], (patch_x + 1) * patch_size[0]
    #         ):
    #             for j in range(
    #                 patch_y * patch_size[1], (patch_y + 1) * patch_size[1]
    #             ):
    #                 if i < image_size[0] and j < image_size[1]:
    #                     s.add((k, b))
    #                     masks[k][b][i, j] = 255
    alter_mask = torch.zeros((2, *image_size), device=device)
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            patch_x = i // patch_size[0]
            patch_y = j // patch_size[1]
            if (patch_x + patch_y) % 2:
                alter_mask[0, i, j] = 1
            else:
                alter_mask[1, i, j] = 1
    masks = alter_mask[:, None, :, :].repeat((1, N, 1, 1))

    blurred_masks = [[None for _ in range(N)] for _ in range(num_masks)]
    for k in range(num_masks):
        for b in range(N):
            mask = transforms.ToPILImage()(masks[k][b])
            blurred_masks[k][b] = transforms.ToTensor()(
                pipeline.mask_processor.blur(mask, blur_factor=blur_factor)
            ).to(device)
    
    for id, blurred_mask in enumerate(blurred_masks):
        blurred_mask_PIL =  transforms.ToPILImage()(blurred_mask)
        blurred_mask_PIL.save(f"blurred_image_{id}.png")
    
        
        
    masks_Pil =  transforms.ToPILImage()(masks)
    images = init_images.clone()
    for mask in blurred_masks:
        tmp = pipeline(
            prompt=["" for _ in range(N)],
            image=images,
            mask_image=mask,
            generator=rng,
        ).images
        for i in range(len(tmp)):
            images[i] = transforms.ToTensor()(tmp[i])
    for id, image in enumerate(images):
        image_PIL =  transforms.ToPILImage()(image)
        image_PIL.save(f"image_{id}.png")
        
    
    return torch.abs(images - init_images)


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
    transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.Resize(args.image_size),
        ]
    )
    image = Image.open(args.image_path).convert('RGB')
    image_tensor = transform(image).to(device)

    print(f"Preprocessed image shape: {image_tensor.shape}")

    if args.float16:
        pipeline = AutoPipelineForInpainting.from_pretrained(
            args.diffuser, torch_dtype=torch.float16, variant="fp16"
        ).to(device)
    else:
        pipeline = AutoPipelineForInpainting.from_pretrained(args.diffuser).to(device)

    # Compute the MRE image
    mre_image_tensor = compute_MRE(
        pipeline=pipeline,
        init_images=image_tensor.unsqueeze(0),
        device=device,
        num_masks=args.num_masks,
        blur_factor=args.blur_factor,
        patch_size=args.patch_size,
        seed=args.seed,
    )
    mre_image_Pil = transforms.ToPILImage()(mre_image_tensor)
    
    mre_image_Pil.save("mre_image.png")
    print(f"save successful")
    print(f"MRE image tensor shape: {mre_image_tensor.shape}")
    

    # Load the trained ResNet model and processor
    processor = AutoImageProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageClassification.from_pretrained(args.model_path, trust_remote_code=True).to(device)

    inputs = processor(images=mre_image_Pil, return_tensors="pt").to(device)
    inputs = {key: val for key, val in inputs.items()}

    print(f"Processed inputs shape: {inputs['pixel_values'].shape}")

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # Calculate probabilities
    probabilities = F.softmax(outputs.logits, dim=1).cpu().numpy()
    real_prob, fake_prob = probabilities[0]

    print(f"Real: {real_prob * 100:.2f}%")
    print(f"Fake: {fake_prob * 100:.2f}%")

if __name__ == "__main__":
    args = create_args()
    main(args)
