import os
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from models import ResNet50Detector

def evaluate_unseen_data():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Görmediği Veri (Unseen Data) Test Cihazı: {device}")

    # Modeli Yükle
    model = ResNet50Detector(freeze_features=False).to(device)
    model.load_state_dict(torch.load("models/resnet50_best.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    real_samples = []
    fake_samples = []

    # Doğrudan data klasörünü tara
    base_dir = "data"
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                if "FAKE" in root.upper() or "AI" in root.upper():
                    fake_samples.append(img_path)
                elif "REAL" in root.upper():
                    real_samples.append(img_path)

    # 100 REAL + 100 FAKE olmak üzere dengeli 200 resimlik test kümesi
    num_samples = min(100, len(real_samples), len(fake_samples))
    test_samples = real_samples[:num_samples] + fake_samples[:num_samples]
    labels = [0] * num_samples + [1] * num_samples

    print(f"🔍 Dengeli Dağılım: {num_samples} REAL + {num_samples} FAKE (Toplam {len(test_samples)} görsel) test ediliyor...")

    preds = []
    with torch.no_grad():
        for img_path in test_samples:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            output = model(tensor)
            _, pred = torch.max(output, 1)
            preds.append(pred.item())

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)

    print("\n========================================================")
    print("🌟 GÖRMEDİĞİ VERİ (UNSEEN DATA) TEST SONUÇLARI")
    print("========================================================")
    print(f"🎯 Accuracy  : %{acc*100:.2f}")
    print(f"🎯 Precision : %{precision*100:.2f}")
    print(f"🎯 Recall    : %{recall*100:.2f}")
    print(f"🎯 F1-Score  : %{f1*100:.2f}")
    print("========================================================\n")

if __name__ == "__main__":
    evaluate_unseen_data()