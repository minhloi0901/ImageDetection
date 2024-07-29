import os
import numpy as np
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from evaluate import load as load_metric
from PIL import Image
from sklearn.model_selection import train_test_split
import wandb
from torchvision import transforms


# Disable wandb
wandb.init(mode="disabled")

class CustomDataset(TorchDataset):
    def __init__(self, images, processor, device):
        self.images = images
        self.processor = processor
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        size = (512, 512)
        img_path= self.images[idx]['image']
        label = self.images[idx]['label']
        image = Image.open(img_path).convert('RGB')
        image = image.resize(size, Image.LANCZOS)

        inputs = self.processor(images=image, return_tensors='pt')
        # Remove extra dimensions and convert tensor to the appropriate shape
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        return {**inputs, 'label': torch.tensor(label)}

def create_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cpu",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed used for training"
    )

    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to initial dataset"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
   
    parser.add_argument(
        "--save-dir", type=str, required=True, help="Where to save the model and logs"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=5, help="Number of epochs"
    )
    return parser.parse_args()

def load_images(image_dir, label, size=(512, 512)):
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_dir, filename)
            images.append({'image': img_path , 'label': label})
    return images

def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    dataset_dir = args.dataset
    real_images_dir = os.path.join(dataset_dir, 'reals')
    fake_images_dir = os.path.join(dataset_dir, 'fakes')

    real_images = load_images(real_images_dir, 0)
    fake_images = load_images(fake_images_dir, 1)
    
    all_images = real_images + fake_images

    # Split dataset with 80% for training and 20% for testing
    train_images, test_images = train_test_split(all_images, test_size=0.2, random_state=args.seed)

    processor = AutoImageProcessor.from_pretrained('microsoft/resnet-50', trust_remote_code=True)
    model = AutoModelForImageClassification.from_pretrained('microsoft/resnet-50', num_labels=2, ignore_mismatched_sizes=True, trust_remote_code=True).to(device)

    train_dataset = CustomDataset(train_images, processor, device)
    test_dataset = CustomDataset(test_images, processor, device)
    
    # Print sizes of train and test datasets
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")

    def compute_metrics(p):
        accuracy = load_metric("accuracy")
        return accuracy.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        run_name='training',
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        save_steps=10_000,
        save_total_limit=2,
        seed=args.seed,
        logging_dir=os.path.join(args.save_dir, 'logs'),  # Directory for logs
        logging_steps=500,  # Log every 500 steps
        learning_rate=5e-5,  # Start with a smaller learning rate
        lr_scheduler_type="cosine",  # Use a cosine scheduler
        load_best_model_at_end=True,  # Load the best model at the end
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    model_save_path = os.path.join(args.save_dir, "model")
    model.save_pretrained(model_save_path)
    processor.save_pretrained(model_save_path)

    results = trainer.evaluate(eval_dataset=test_dataset)

    with open(os.path.join(args.save_dir, "test_results.txt"), "w") as writer:
        for key, value in results.items():
            writer.write(f"{key}: {value}\n")
    # Extract misclassified examples
    misclassified_indices = results.get('misclassified_idx', [])
    
    # Load misclassified images
    misclassified_images = [test_images[idx]['image'] for idx in misclassified_indices]
    misclassified_labels = [test_images[idx]['label'] for idx in misclassified_indices]

    # Display misclassified images
    for i, (img, label) in enumerate(zip(misclassified_images, misclassified_labels)):
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(f"Misclassified Image {i + 1}, True Label: {label}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    args = create_args()
    main(args)
#fdas