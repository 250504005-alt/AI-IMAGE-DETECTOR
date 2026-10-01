import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from dataset import get_dataloaders
from models import ResNet50Detector

def train_resnet():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 ResNet50 Eğitim Cihazı: {device}")

    # Veri yükleyicileri (img_size=128, batch_size=128)
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir="data", batch_size=128, num_workers=0, img_size=128
    )

    model = ResNet50Detector(freeze_features=True).to(device)
    criterion = nn.CrossEntropyLoss()

    # ==========================================
    # 1. AŞAMA: TRANSFER LEARNING (Sadece Son Katman)
    # ==========================================
    optimizer = optim.Adam(model.resnet.fc.parameters(), lr=0.001)
    epochs = 3
    print("\n--- 🏋️ RESNET50 TRANSFER LEARNING (Aşama 1: Sadece Son Katman) ---")
    
    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == total_batches:
                print(f"Epoch [{epoch+1}/{epochs}] | Adım [{batch_idx+1}/{total_batches}] - Loss: {loss.item():.4f}")

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"✨ Epoch [{epoch+1}/{epochs}] Özet -> Train Loss: {epoch_train_loss:.4f} | Val Acc: {val_acc:.4f}\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/resnet50_best.pth")

    # ==========================================
    # 2. AŞAMA: FINE-TUNING (Layer4 Kilidi Açılıyor)
    # ==========================================
    print("\n--- 🔓 RESNET50 FINE-TUNING (Aşama 2: Layer4 Kilidi Açılıyor) ---")
    model.load_state_dict(torch.load("models/resnet50_best.pth"))
    
    # Layer4 katmanının kilidini aç
    for param in model.resnet.layer4.parameters():
        param.requires_grad = True

    # Düşük öğrenme oranı (1e-4) ile ince ayar
    optimizer_ft = optim.Adam([
        {'params': model.resnet.layer4.parameters(), 'lr': 1e-4},
        {'params': model.resnet.fc.parameters(), 'lr': 1e-4}
    ])

    ft_epochs = 3
    for epoch in range(ft_epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer_ft.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_ft.step()
            running_loss += loss.item() * images.size(0)

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == total_batches:
                print(f"Fine-Tune Epoch [{epoch+1}/{ft_epochs}] | Adım [{batch_idx+1}/{total_batches}] - Loss: {loss.item():.4f}")

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"🌟 Fine-Tune Epoch [{epoch+1}/{ft_epochs}] Özet -> Train Loss: {epoch_train_loss:.4f} | Val Acc: {val_acc:.4f}\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/resnet50_best.pth")

    # ==========================================
    # 3. AŞAMA: FINAL TEST DEĞERLENDİRMESİ VE GRAFİK KAYDI
    # ==========================================
    print("\n--- 📊 RESNET50 FINAL TEST DEĞERLENDİRMESİ ---")
    model.load_state_dict(torch.load("models/resnet50_best.pth"))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

    print(f"🎯 ResNet50 Test Accuracy  : {acc:.4f}")
    print(f"🎯 ResNet50 Test Precision : {precision:.4f}")
    print(f"🎯 ResNet50 Test Recall    : {recall:.4f}")
    print(f"🎯 ResNet50 Test F1-Score  : {f1:.4f}")

    # Confusion Matrix Çizimi & Kaydı
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL', 'AI/FAKE'], yticklabels=['REAL', 'AI/FAKE'])
    plt.title('ResNet50 - Confusion Matrix')
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    os.makedirs('results/graphs', exist_ok=True)
    plt.savefig('results/graphs/resnet50_confusion_matrix.png')
    plt.close()

    print("🖼️ ResNet50 Confusion Matrix grafiği 'results/graphs/resnet50_confusion_matrix.png' olarak kaydedildi!")

if __name__ == '__main__':
    train_resnet()