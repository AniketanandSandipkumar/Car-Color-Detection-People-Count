import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Car Color & People Count", layout="centered")
st.title("🚗 Car Color Detection & 👥 People Count")
st.write("Detect cars and people using YOLOv8 and apply simple color-based logic.")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Nano model (lightweight)

model = load_model()

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

def detect_car_color(car_roi):
    if car_roi.size == 0:
        return "Unknown"

    avg_color = np.mean(car_roi, axis=(0, 1))  # BGR
    b, g, r = avg_color

    if b > r and b > g:
        return "Blue"
    elif r > b and r > g:
        return "Red"
    elif g > r and g > b:
        return "Green"
    else:
        return "Other"

if uploaded_image:
    image = Image.open(uploaded_image)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = model(frame, conf=0.4)

    car_count = 0
    person_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            roi = frame[y1:y2, x1:x2]

            # 🚗 Car class (COCO = 2)
            if cls == 2:
                car_count += 1
                car_color = detect_car_color(roi)

                box_color = (255, 0, 0) if car_color == "Blue" else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(
                    frame,
                    f"Car ({car_color})",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2,
                )

            # 👥 Person class (COCO = 0)
            elif cls == 0:
                person_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "Person",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    st.image(frame, caption="Detection Result", use_container_width=True)

    st.subheader("📊 Detection Summary")
    st.write(f"🚗 Cars detected: {car_count}")
    st.write(f"👥 People detected: {person_count}")