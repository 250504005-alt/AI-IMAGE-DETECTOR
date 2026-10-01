import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import sys
import os

# src modüllerini görebilmesi için yol ekleme
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from models import ResNet50Detector
from gradcam import GradCAM

st.set_page_config(page_title="AI Image Detector", page_icon="🖼️", layout="centered")

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = ResNet50Detector(freeze_features=False).to(device)
    model_path = "models/resnet50_best.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

try:
    model, device = load_model()
    target_layer = model.resnet.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")
    st.stop()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.markdown("<h1 style='text-align: center;'>AI IMAGE DETECTOR</h1>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("Upload Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', width='stretch')

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        real_prob = probabilities[0].item() * 100
        ai_prob = probabilities[1].item() * 100
        _, predicted = torch.max(outputs, 1)

    prediction_text = "AI Generated" if predicted.item() == 1 else "Real"
    color = "red" if predicted.item() == 1 else "green"

    st.markdown(f"<h3 style='text-align: center;'>Prediction: <span style='color:{color};'>{prediction_text}</span></h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="AI Probability", value=f"%{ai_prob:.1f}")
    with col2:
        st.metric(label="Real Probability", value=f"%{real_prob:.1f}")

    st.info("ℹ️ *These probabilities represent the model's confidence, not the actual percentage of AI content in the image.*")

    st.markdown("---")
    
    st.markdown("<h3 style='text-align: center;'>Grad-CAM Analysis 🔥</h3>", unsafe_allow_html=True)
    
    heatmap, _ = grad_cam.generate_heatmap(input_tensor)
    
    img_resized = image.resize((128, 128))
    img_np = np.array(img_resized)
    
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
    
    st.image(overlay, caption='Focus Areas (Heatmap)', width='stretch')