import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load model once at the start
@st.cache_resource
def load_model():
    model = YOLO("runs/classify/train/weights/best.pt")
    return model

model = load_model()

# UI Layout
st.set_page_config(page_title="Kidney Disease Classifier", layout="centered")
st.title("🧠 Kidney Disease Classifier (YOLOv8)")
st.caption("Upload a CT scan image of a kidney to classify it as Normal, Tumor, Cyst, or Stone.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload a CT Image", type=["jpg", "jpeg", "png"])

# On image upload
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Analyzing Image..."):
        results = model.predict(image, imgsz=224, device=0)  # Use GPU (device=0)

        pred_idx = results[0].probs.top1
        confidence = results[0].probs.top1conf
        label_map = results[0].names
        prediction = label_map[pred_idx]

    st.success(f"✅ **Prediction:** `{prediction}` with **{confidence * 100:.2f}%** confidence")

    # Show full probability breakdown
    st.subheader("📊 Class Probabilities")
    for i, prob in enumerate(results[0].probs.data.tolist()):
        st.write(f"- **{label_map[i]}**: {prob*100:.2f}%")
