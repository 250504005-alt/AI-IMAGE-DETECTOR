import os
import glob
import matplotlib.pyplot as plt
from PIL import Image

def run_eda(data_dir="data"):
    print("--- 📊 KEŞİFSEL VERİ ANALİZİ (EDA) ---")
    
    # Tüm data klasörü altındaki REAL ve FAKE görsellerini bul
    real_imgs = glob.glob(os.path.join(data_dir, "**", "REAL", "*.jpg"), recursive=True) + \
                glob.glob(os.path.join(data_dir, "**", "REAL", "*.png"), recursive=True)
    
    fake_imgs = glob.glob(os.path.join(data_dir, "**", "FAKE", "*.jpg"), recursive=True) + \
                glob.glob(os.path.join(data_dir, "**", "FAKE", "*.png"), recursive=True)

    print(f"[DATA] Bulunan Toplam REAL Görsel: {len(real_imgs)} | Toplam AI FAKE Görsel: {len(fake_imgs)}")

    if len(real_imgs) < 16 or len(fake_imgs) < 16:
        print("⚠️ Görsel sayısı henüz yetersiz veya klasör yapısı taranıyor.")
        return

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle('CIFAKE Görsel Örnekleri (Sol 16: REAL | Sağ 16: AI FAKE)', fontsize=16)

    # REAL Örnekleri (Sol 4 sütun)
    for idx in range(16):
        row, col = idx // 4, idx % 4
        img = Image.open(real_imgs[idx])
        axes[row, col].imshow(img)
        axes[row, col].set_title('REAL', color='green', fontsize=9)
        axes[row, col].axis('off')

    # FAKE Örnekleri (Sağ 4 sütun)
    for idx in range(16):
        row, col = idx // 4, (idx % 4) + 4
        img = Image.open(fake_imgs[idx])
        axes[row, col].imshow(img)
        axes[row, col].set_title('AI FAKE', color='red', fontsize=9)
        axes[row, col].axis('off')

    plt.tight_layout()
    os.makedirs('results/graphs', exist_ok=True)
    plt.savefig('results/graphs/eda_samples.png')
    print("✅ Örnek görseller 'results/graphs/eda_samples.png' olarak başarıyla kaydedildi!")
    plt.show()

if __name__ == '__main__':
    run_eda()