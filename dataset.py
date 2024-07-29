import os

from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root: str, transform = None):
        self.root = root
        self.samples = []
        self.transform = transform
        self.classes = {}

        for dir in sorted(
            os.scandir(self.root),
            key=lambda x: x.name,
        ):
            if not dir.is_dir():
                continue

            for file in sorted(
                os.scandir(dir.path),
                key=lambda x: x.name,
            ):
                self.samples.append((file.path, len(self.classes)))

            self.classes[len(self.classes)] = dir.name

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")  
        if self.transform is not None:
            image = self.transform(image) 

        return image, label
