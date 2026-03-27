import os
os.system("pip uninstall -y opencv-python")

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Car & People Detection", layout="centered")

st.title("🚗 Car Color Detection & 👥 People Counter")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Simple color detection
def get_color(image, box):
    x1, y1, x2, y2 = map(int, box)
    crop = image[y1:y2, x1:x2]

    avg_color = np.mean(crop, axis=(0, 1))

    if avg_color[2] > 150:
        return "Red"
    elif avg_color[1] > 150:
        return "Green"
    elif avg_color[0] > 150:
        return "Blue"
    else:
        return "Dark"

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    results = model(image_np)

    people_count = 0
    car_count = 0

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0]

            label = model.names[cls]

            if label == "person":
                people_count += 1
                color = (0, 255, 0)

            elif label == "car":
                car_count += 1
                detected_color = get_color(image_np, (x1, y1, x2, y2))
                label = f"Car ({detected_color})"
                color = (255, 0, 0)

            else:
                continue

            cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image_np, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    st.image(image_np, caption="Processed Image", use_column_width=True)

    st.success(f"👥 People Count: {people_count}")
    st.success(f"🚗 Car Count: {car_count}")
