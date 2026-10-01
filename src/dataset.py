import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class CIFAKEDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(data_dir="data", batch_size=128, num_workers=0, img_size=32):
    real_imgs = glob.glob(os.path.join(data_dir, "**", "REAL", "*.jpg"), recursive=True) + \
                glob.glob(os.path.join(data_dir, "**", "REAL", "*.png"), recursive=True)
    
    fake_imgs = glob.glob(os.path.join(data_dir, "**", "FAKE", "*.jpg"), recursive=True) + \
                glob.glob(os.path.join(data_dir, "**", "FAKE", "*.png"), recursive=True)

    all_paths = real_imgs + fake_imgs
    all_labels = [0] * len(real_imgs) + [1] * len(fake_imgs)

    # İstenen img_size değerine göre resize ekliyoruz
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = CIFAKEDataset(all_paths, all_labels, transform=transform)

    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader