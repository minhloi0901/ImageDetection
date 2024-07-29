import os
from argparse import ArgumentParser

from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from diffusers.pipelines.auto_pipeline import AutoPipelineForInpainting
from datasets import load_dataset
from huggingface_hub import login

from PIL import Image

from dataset import ImageDataset


def create_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cpu",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed used for Generator"
    )
    parser.add_argument(
        "--hf-token", type=str, default=None, help="Hugging face access token"
    )

    # Datasets
    parser.add_argument(
        "--num-samples",
        type=int,
        required=True,
        help="Number of samples in dataset to download for each labels",
    )
    parser.add_argument(
        "--root", type=str, required=True, help="Path to initial dataset"
    )
    parser.add_argument(
        "--real-dir",
        type=str,
        nargs="+",
        default=None,
        help="Path to real dataset",
    )
    parser.add_argument(
        "--fake-dir",
        type=str,
        nargs="+",
        default=None,
        help="Path to fake dataset",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        default=[512, 512],
        help="Size of images",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
    )
    parser.add_argument("--num-workers", type=int, default=1)

    # Diffuser
    parser.add_argument(
        "--diffuser",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
        help="Diffuser to used",
    )
    parser.add_argument(
        "--float16",
        action="store_true",
    )

    # MRE
    parser.add_argument(
        "--save-dir", type=str, required=True, help="Where to save"
    )
    parser.add_argument(
        "--num-masks", type=int, default=2, help="Number of masks"
    )
    parser.add_argument(
        "--blur-factor",
        type=float,
        default=10,
        help="Blur factor used for blurring masks",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs="+",
        default=[64, 64],
        help="Size of patch used for computing MRE",
    )
    return parser.parse_args()


# MRE = masked restoration error


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

    return torch.abs(images - init_images)


def main(args):
    if args.hf_token:
        login(token=args.hf_token)
    args.device = torch.device(args.device)

    if args.real_dir:
        hf_dataset = load_dataset(
            *args.real_dir,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        hf_iter = iter(hf_dataset)
        os.makedirs(os.path.join(args.root, "reals"), exist_ok=True)
        for _ in range(1000):
            next(hf_iter)
        for i in tqdm(range(args.num_samples), desc="Downloading real dataset"):
            r = next(hf_iter)
            os.makedirs(os.path.join(args.root, "reals"), exist_ok=True)
            r["image"].save(os.path.join(args.root, "reals", f"{i + 1000}.png"))

    if args.fake_dir:
        hf_dataset = load_dataset(
            *args.fake_dir,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        hf_iter = iter(hf_dataset)
        os.makedirs(os.path.join(args.root, "fakes"), exist_ok=True)
        for _ in range(1000):
            next(hf_iter)
        for i in tqdm(range(args.num_samples), desc="Downloading fake dataset"):
            r = next(hf_iter)
            r["image"].save(os.path.join(args.root, "fakes", f"{i + 1000}.png"))

    transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.Resize(args.image_size),
        ]
    )
    dataset = ImageDataset(root=args.root, transform=transform)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    if args.float16:
        pipeline = AutoPipelineForInpainting.from_pretrained(
            args.diffuser, torch_dtype=torch.float16, variant="fp16"
        ).to(args.device)
    else:
        pipeline = AutoPipelineForInpainting.from_pretrained(args.diffuser).to(
            args.device
        )

    os.makedirs(args.save_dir, exist_ok=True)
    cnt = 0
    for images, labels in dataloader:
        images = images.to(args.device)
        mre_images = compute_MRE(
            pipeline=pipeline,
            init_images=images,
            device=args.device,
            num_masks=args.num_masks,
            blur_factor=args.blur_factor,
            patch_size=args.patch_size,
            seed=args.seed,
        )

        for i in range(len(labels)):
            label_dir = os.path.join(args.save_dir, dataset.classes[labels[i].item()])

            os.makedirs(
                label_dir,
                exist_ok=True,
            )
            pil_image = transforms.ToPILImage()(mre_images[i])
            pil_image.save(os.path.join(label_dir, f"{cnt}.png"))
            cnt += 1


if __name__ == "__main__":
    args = create_args()
    main(args)
