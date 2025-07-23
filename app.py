import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
from ultralytics import YOLO

# ⚙️ Common class labels
CLASS_NAMES = ['Cyst', 'Normal', 'Tumor', 'Stone']

# 🔧 Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ✅ Load all models once
@st.cache_resource
def load_models():
    models_dict = {}

    # YOLOv6 - Custom RepVGG
    from YOLOv6.yolov6.models.repvgg import RepVGG
    model_v6 = RepVGG(num_classes=4)
    model_v6.load_state_dict(torch.load("v6_classifier.pth", map_location="cpu"))
    model_v6.eval()
    models_dict['YOLOv6'] = model_v6
    
    # YOLOv8
    model_v8 = YOLO("runs/classify/train/weights/best.pt")
    models_dict['YOLOv8'] = model_v8

    # YOLOv7 - ResNet18
    model_v7 = models.resnet18(num_classes=4)
    model_v7.load_state_dict(torch.load("kidney_classifier_v7.pth", map_location="cpu"))
    model_v7.eval()
    models_dict['YOLOv7'] = model_v8


    return models_dict

# 🌐 UI
st.set_page_config(page_title="Kidney Disease Classifier", layout="centered")
st.title("🧠 Kidney Disease Classifier")
st.caption("Upload a CT scan image of a kidney to classify it as Normal, Tumor, Cyst, or Stone.")

# 🔘 Model selector
model_choice = st.radio("🧬 Select YOLO Version:", ['YOLOv6', 'YOLOv7', 'YOLOv8'], horizontal=True)

# 📥 File upload
uploaded_file = st.file_uploader("📤 Upload a CT Image", type=["jpg", "jpeg", "png"])
models_loaded = load_models()

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner(f"🔍 Analyzing Image using {model_choice}..."):
        if model_choice == 'YOLOv8' or model_choice == 'YOLOv7':
            results = models_loaded['YOLOv8'].predict(image, imgsz=224, device=0)[0]
            pred_idx = results.probs.top1
            confidence = results.probs.top1conf
            prediction = results.names[pred_idx]
            st.success(f"✅ **Prediction:** `{prediction}` with **{confidence * 100:.2f}%** confidence")
            st.subheader("📊 Class Probabilities")
            for i, prob in enumerate(results.probs.data.tolist()):
                st.write(f"- **{CLASS_NAMES[i]}**: {prob * 100:.2f}%")

        else:
            image_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = models_loaded[model_choice](image_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                pred_idx = torch.argmax(probs).item()
                prediction = CLASS_NAMES[pred_idx]
                confidence = probs[pred_idx].item()
                st.success(f"✅ **Prediction:** `{prediction}` with **{confidence * 100:.2f}%** confidence")
                st.subheader("📊 Class Probabilities")
                for i, prob in enumerate(probs.tolist()):
                    st.write(f"- **{CLASS_NAMES[i]}**: {prob * 100:.2f}%")
