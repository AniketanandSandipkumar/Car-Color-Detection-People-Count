import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Car Color & People Count", layout="centered")
st.title("🚗 Car Color Detection & 👥 People Count")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

def detect_car_color(car_roi):
    if car_roi.size == 0:
        return "Unknown"

    avg_color = np.mean(car_roi, axis=(0, 1))  # RGB
    r, g, b = avg_color

    if b > r and b > g:
        return "Blue"
    elif r > g and r > b:
        return "Red"
    elif g > r and g > b:
        return "Green"
    else:
        return "Other"

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    frame = np.array(image)

    results = model(frame, conf=0.4)

    car_count = 0
    person_count = 0

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            roi = frame[y1:y2, x1:x2]

            if cls == 2:  # Car
                car_count += 1
                car_color = detect_car_color(roi)

            elif cls == 0:  # Person
                person_count += 1

        # Use YOLO built-in drawing
        plotted = r.plot()

    st.image(plotted, caption="Detection Result", use_container_width=True)

    st.subheader("📊 Detection Summary")
    st.write(f"🚗 Cars detected: {car_count}")
    st.write(f"👥 People detected: {person_count}")
