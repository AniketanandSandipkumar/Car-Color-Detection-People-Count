import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

st.title("🚗 Car & 👤 People Detection App")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    results = model(img_array)

    draw = ImageDraw.Draw(image)

    person_count = 0
    car_count = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            if label in ["person", "car"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label == "person":
                    person_count += 1
                    color = "red"
                else:
                    car_count += 1
                    color = "blue"

                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.text((x1, y1), f"{label} {conf:.2f}", fill=color)

    st.image(image, caption="Detection Result", use_column_width=True)

    st.subheader("📊 Detection Counts")
    st.write(f"👤 People: {person_count}")
    st.write(f"🚗 Cars: {car_count}")

