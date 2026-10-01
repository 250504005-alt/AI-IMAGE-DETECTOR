import os
import torch
import numpy as np
import cv2
from PIL import Image, ImageFilter
from torchvision import transforms
from sklearn.metrics import accuracy_score

from dataset import get_dataloaders
from models import ResNet50Detector

def apply_jpeg_compression(img_pil, quality=75):
    """Görsele JPEG sıkıştırması uygular."""
    buffer = torch.hub.io.BytesIO() if hasattr(torch.hub, 'io') else __import__('io').BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

def apply_blur(img_pil, radius=2):
    """Görsele Gaussian Blur uygular."""
    return img_pil.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_resize_degrade(img_pil, low_size=(32, 32)):
    """Görselin boyutunu önce düşürüp sonra tekrar yükselterek kaliteyi bozar."""
    orig_size = img_pil.size
    low_img = img_pil.resize(low_size, Image.BILINEAR)
    return low_img.resize(orig_size, Image.BILINEAR)

def evaluate_robustness():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Dayanıklılık (Robustness) Test Cihazı: {device}")

    # Test veri yükleyici
    _, _, test_loader = get_dataloaders(data_dir="data", batch_size=128, num_workers=0, img_size=128)

    # Modeli Yükle
    model = ResNet50Detector(freeze_features=False).to(device)
    model.load_state_dict(torch.load("models/resnet50_best.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_scenarios = [
        ("Orijinal Test Seti", None),
        ("JPEG Quality 75", lambda img: apply_jpeg_compression(img, 75)),
        ("JPEG Quality 50", lambda img: apply_jpeg_compression(img, 50)),
        ("Gaussian Blur (r=2)", lambda img: apply_blur(img, radius=2)),
        ("Low-Res Degradation (32x32 -> 128x128)", lambda img: apply_resize_degrade(img, (32, 32)))
    ]

    print("\n========================================================")
    print("🧪 DAYANIKLILIK (ROBUSTNESS) TESTLERİ BAŞLIYOR")
    print("========================================================\n")

    # Bütün test kümesinden ilk 1000 örnek üzerinde hızlı ve hassas test yapıyoruz
    all_raw_images = []
    all_labels = []

    count = 0
    # DataLoader'dan tensörleri alıp PIL görsele dönüştürerek saklayalım
    unnormalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    for images, labels in test_loader:
        for i in range(images.size(0)):
            img_tensor = unnormalize(images[i])
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            all_raw_images.append(img_pil)
            all_labels.append(labels[i].item())
            count += 1
            if count >= 1000:  # 1000 görsellik temsilci test kümesi
                break
        if count >= 1000:
            break

    for scenario_name, scenario_fn in test_scenarios:
        preds = []
        with torch.no_grad():
            for i, img_pil in enumerate(all_raw_images):
                processed_img = scenario_fn(img_pil) if scenario_fn else img_pil
                input_tensor = transform(processed_img).unsqueeze(0).to(device)
                outputs = model(input_tensor)
                _, pred = torch.max(outputs, 1)
                preds.append(pred.item())

        acc = accuracy_score(all_labels, preds)
        print(f"📊 {scenario_name:<40} -> Accuracy: %{acc*100:.2f}")

    print("\n✅ Dayanıklılık testleri başarıyla tamamlandı!")

if __name__ == "__main__":
    evaluate_robustness()