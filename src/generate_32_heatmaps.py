import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from models import ResNet50Detector
from gradcam import GradCAM

def generate_32_heatmaps():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 32 Heatmap Analiz Cihazı: {device}")

    # Modeli Yükle
    model = ResNet50Detector(freeze_features=False).to(device)
    model.load_state_dict(torch.load("models/resnet50_best.pth", map_location=device))
    model.eval()

    target_layer = model.resnet.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Kümeler
    categories = {
        "REAL_Correct": [],
        "AI_Correct": [],
        "REAL_Wrong": [],
        "AI_Wrong": []
    }

    base_dir = "data/test" if os.path.exists("data/test") else "data"
    print("🔍 Test verileri taranıyor ve 4 grup oluşturuluyor...")

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                true_label = 1 if ("FAKE" in root.upper() or "AI" in root.upper()) else 0

                raw_img = Image.open(img_path).convert("RGB")
                tensor = transform(raw_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(tensor)
                    _, pred = torch.max(output, 1)
                    pred_label = pred.item()

                # Gruplama
                if true_label == 0 and pred_label == 0 and len(categories["REAL_Correct"]) < 8:
                    categories["REAL_Correct"].append((img_path, raw_img, tensor))
                elif true_label == 1 and pred_label == 1 and len(categories["AI_Correct"]) < 8:
                    categories["AI_Correct"].append((img_path, raw_img, tensor))
                elif true_label == 0 and pred_label == 1 and len(categories["REAL_Wrong"]) < 8:
                    categories["REAL_Wrong"].append((img_path, raw_img, tensor))
                elif true_label == 1 and pred_label == 0 and len(categories["AI_Wrong"]) < 8:
                    categories["AI_Wrong"].append((img_path, raw_img, tensor))

                # Her gruptan 8 resim tamamlandıysa döngüden çık
                if all(len(v) >= 8 for v in categories.values()):
                    break

    output_dir = "results/graphs/heatmaps_32"
    os.makedirs(output_dir, exist_ok=True)

    print("\n🔥 Heatmap'ler üretiliyor ve kaydediliyor...")

    count = 0
    for cat_name, samples in categories.items():
        for idx, (img_path, raw_img, input_tensor) in enumerate(samples):
            heatmap, pred_class = grad_cam.generate_heatmap(input_tensor)

            raw_resized = raw_img.resize((128, 128))
            img_np = np.array(raw_resized)

            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            axes[0].imshow(img_np)
            axes[0].set_title("Orijinal")
            axes[0].axis("off")

            axes[1].imshow(heatmap, cmap="jet")
            axes[1].set_title("Grad-CAM")
            axes[1].axis("off")

            axes[2].imshow(overlay)
            axes[2].set_title(f"Grup: {cat_name}")
            axes[2].axis("off")

            save_path = os.path.join(output_dir, f"{cat_name}_{idx+1}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            count += 1

    print(f"✅ Toplam {count} adet Grad-CAM ısı haritası '{output_dir}' klasörüne kaydedildi!")

if __name__ == "__main__":
    generate_32_heatmaps()