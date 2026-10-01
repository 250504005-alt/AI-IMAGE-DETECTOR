import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from models import ResNet50Detector

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook tanımlamaları
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (128, 128))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam, target_class

def run_gradcam_demo():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Modeli yükle
    model = ResNet50Detector(freeze_features=False).to(device)
    model.load_state_dict(torch.load("models/resnet50_best.pth", map_location=device))
    
    # ResNet50 için hedef katman: layer4
    target_layer = model.resnet.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    # Test klasöründen rastgele bir resim seçelim
    test_dir = "data"
    fake_images = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if "FAKE" in root and (file.endswith(".jpg") or file.endswith(".png")):
                fake_images.append(os.path.join(root, file))

    if not fake_images:
        print("❌ Test edilecek görsel bulunamadı.")
        return

    sample_img_path = fake_images[0]
    print(f"🖼️ Seçilen Test Görseli: {sample_img_path}")

    # Görsel Ön İşleme
    raw_img = Image.open(sample_img_path).convert("RGB").resize((128, 128))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    # Heatmap Oluşturma
    heatmap, pred_class = grad_cam.generate_heatmap(input_tensor)
    
    # Heatmap Görselleştirme
    img_np = np.array(raw_img)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

    # Kaydetme
    os.makedirs("results/graphs", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    labels = ["REAL", "AI / FAKE"]
    
    axes[0].imshow(img_np)
    axes[0].set_title("Orijinal Görsel")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Isı Haritası")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Tahmin: {labels[pred_class]}")
    axes[2].axis("off")

    plt.savefig("results/graphs/gradcam_result.png", bbox_inches='tight')
    plt.close()
    
    print("🔥 Grad-CAM analizi tamamlandı! Sonuç 'results/graphs/gradcam_result.png' adresine kaydedildi.")

if __name__ == "__main__":
    run_gradcam_demo()