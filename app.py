import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Load trained model (update path to your custom model if needed)
model = YOLO("runs/detect/train5/weights/best.pt")  # e.g., "yolov8_car_damage.pt" for custom model

# Streamlit UI
st.set_page_config(page_title="Car Damage Detection", layout="centered")
st.title("🚗 Car Damage Detection System")
st.write("Upload a car image, and the model will detect damaged parts using YOLOv8.")

# File uploader
uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Get file extension (e.g., .jpg)
    file_ext = os.path.splitext(uploaded_file.name)[1]
    
    # Create temporary file with the correct extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Display original image
    image = Image.open(temp_file_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Inference with YOLOv8
    st.info("🔍 Detecting car damage...")
    results = model(temp_file_path)

    # Display result image with detections
    result_image = results[0].plot()
    st.image(result_image, caption="Detected Damage", use_column_width=True)
    st.success("✅ Detection complete!")

    # Show detected labels and confidence scores
    st.subheader("Detection Summary:")
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            st.write(f"🟠 `{label}` detected with {conf:.2%} confidence")

    # Clean up
    os.remove(temp_file_path)
