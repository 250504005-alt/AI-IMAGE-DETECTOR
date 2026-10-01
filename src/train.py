import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from dataset import get_dataloaders
from models import BaselineCNN

def train_model():
    # 1. Cihaz Seçimi (Apple Silicon MPS Acceleration)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Eğitim Cihazı: {device}")

    # 2. Veri Yükleyicileri (num_workers=0 eklendi)
    train_loader, val_loader, test_loader = get_dataloaders(data_dir="data", batch_size=128, num_workers=0)

    # 3. Model, Loss ve Optimizer
    model = BaselineCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    train_losses, val_losses = [], []

    print("\n--- 🏋️ BASELINE CNN EĞİTİMİ BAŞLIYOR (10 Epoch) ---")
    
    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/baseline_cnn.pth")

    print("\n✅ Eğitim Tamamlandı! En iyi model kaydedildi: models/baseline_cnn.pth")

    # Loss Grafiği
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Baseline CNN - Loss Eğrisi')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    os.makedirs('results/graphs', exist_ok=True)
    plt.savefig('results/graphs/cnn_loss.png')
    plt.close()

    # TEST DEĞERLENDİRMESİ
    print("\n--- 📊 TEST KÜMESİ DEĞERLENDİRMESİ ---")
    model.load_state_dict(torch.load("models/baseline_cnn.pth"))
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

    print(f"🎯 Test Accuracy  : {acc:.4f}")
    print(f"🎯 Test Precision : {precision:.4f}")
    print(f"🎯 Test Recall    : {recall:.4f}")
    print(f"🎯 Test F1-Score  : {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL', 'AI/FAKE'], yticklabels=['REAL', 'AI/FAKE'])
    plt.title('Baseline CNN - Confusion Matrix')
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.savefig('results/graphs/cnn_confusion_matrix.png')
    plt.close()

    print("🖼️ Grafik çıktılan kaydedildi!")

if __name__ == '__main__':
    train_model()