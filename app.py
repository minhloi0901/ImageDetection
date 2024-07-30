from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from diffusers.pipelines.auto_pipeline import AutoPipelineForInpainting
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F
import io

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./model"
processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForImageClassification.from_pretrained(model_path, trust_remote_code=True).to(device)
# Load Diffuser
pipeline = AutoPipelineForInpainting.from_pretrained("runwayml/stable-diffusion-inpainting").to(device)
   
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
    
    # for id, blurred_mask in enumerate(blurred_masks):
    #     blurred_mask_PIL =  transforms.ToPILImage()(blurred_mask[0])
    #     blurred_mask_PIL.save(f"blurred_image_{id}.png")
    
        
        
    images = init_images.clone()
    for id, image in enumerate(images):
        image_PIL =  transforms.ToPILImage()(image)
        # image_PIL.save(f"init_image_{k}_{id}.png")
    for k, mask in enumerate(blurred_masks):
        tmp = pipeline(
            prompt=["" for _ in range(N)],
            image=images,
            mask_image=mask,
            generator=rng,
        ).images
        
        for i in range(len(tmp)):
            images[i] = transforms.ToTensor()(tmp[i])
        
    
    return torch.abs(images - init_images)

@app.route('/predict', methods=['POST'])
def predict():
    print("call API")
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')
    image.save("fromFronent.png")
    # Preprocess the image
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize((512, 512)),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
     
    # Compute MRE
    mre_image_tensor = compute_MRE(
        pipeline=pipeline,
        init_images=image_tensor,
        device=device,
        num_masks=2,  # Adjust parameters as needed
        blur_factor=10,
        patch_size=(64, 64),
        seed=0,
    )
    mre_image_Pil = transforms.ToPILImage()(mre_image_tensor[0])
    
    # Prepare the image for the classifier
    inputs = processor(images=mre_image_Pil, return_tensors="pt").to(device)
    inputs = {key: val for key, val in inputs.items()}
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = F.softmax(outputs.logits, dim=1).cpu().numpy()
    real_prob, fake_prob = probabilities[0]
    print(real_prob, fake_prob)
    return jsonify({
        'real_prob': float(real_prob * 100),
        'fake_prob': float(fake_prob * 100)
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6789, debug=True)
