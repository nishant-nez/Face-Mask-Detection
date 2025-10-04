import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import time
import os
import gdown
import requests

# Page configuration
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Model download function
@st.cache_resource
def download_model():
    model_path = 'face_mask_detector.h5'
    
    if not os.path.exists(model_path):
        with st.spinner('Downloading model... This may take a minute.'):
            try:
                # Option A: Google Drive (recommended)
                # Replace 'YOUR_FILE_ID' with your actual Google Drive file ID
                file_id = '1Ms4mpeBdXNIV7U01XHK4vkeO2SrANoGV'
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, model_path, quiet=False)
                
                st.success('Model downloaded successfully!')
            except Exception as e:
                st.error(f'Error downloading model: {e}')
                st.stop()
    
    return model_path

# Load model
@st.cache_resource
def load_mask_model():
    model_path = download_model()
    model = load_model(model_path)
    return model

@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model = load_mask_model()
face_cascade = load_face_detector()

# Helper functions
def detect_face(img):
    faces = face_cascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces

def detect_mask(img):
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    img_reshaped = img_normalized.reshape(1, 224, 224, 3)
    
    prediction = model.predict(img_reshaped, verbose=0)
    probability = prediction[0][0]
    
    return probability

def draw_label(img, text, pos, bg_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    
    text_x = pos[0]
    text_y = pos[1]
    box_coords = (
        (text_x, text_y + 5),
        (text_x + text_size[0] + 10, text_y - text_size[1] - 10)
    )
    
    cv2.rectangle(img, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    cv2.putText(
        img, text, (text_x + 5, text_y),
        font, font_scale, (255, 255, 255), font_thickness
    )

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_face(gray)
    
    mask_detected = False
    no_mask_detected = False
    
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        prediction = detect_mask(face_img)
        
        if prediction < 0.5:
            label = "Mask On"
            color = (0, 255, 0)  # Green
            mask_detected = True
        else:
            label = "No Mask"
            color = (0, 0, 255)  # Red
            no_mask_detected = True
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        draw_label(frame, label, (x, y-10), color)
    
    return frame, mask_detected, no_mask_detected, len(faces)

# Header
st.markdown("""
    <div class="header">
        <h1>😷 Face Mask Detection System</h1>
        <p>Real-time face mask detection using AI</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    detection_mode = st.radio(
        "Detection Mode",
        ["Webcam", "Upload Image"],
        help="Choose between real-time webcam or image upload"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    stats_placeholder = st.empty()
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    This application uses a deep learning model based on VGG16 
    to detect whether a person is wearing a face mask.
    
    **Model Accuracy**: ~94%
    """)
    
    st.markdown("---")
    st.markdown("### 🛡️ Safety Guidelines")
    st.warning("""
    - Wear your mask properly
    - Cover nose and mouth
    - Maintain social distance
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    if detection_mode == "Webcam":
        st.markdown("### 📹 Live Webcam Feed")
        
        run_webcam = st.checkbox("Start Webcam", value=False)
        frame_placeholder = st.empty()
        
        if run_webcam:
            cap = cv2.VideoCapture("http://192.168.1.67:8080/video")
            
            total_frames = 0
            mask_count = 0
            no_mask_count = 0
            
            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam")
                    break
                
                processed_frame, mask_det, no_mask_det, face_count = process_frame(frame)
                
                # Update statistics
                total_frames += 1
                if mask_det:
                    mask_count += 1
                if no_mask_det:
                    no_mask_count += 1
                
                # Convert BGR to RGB for display
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)
                
                # Update stats in sidebar
                with stats_placeholder.container():
                    st.metric("Faces Detected", face_count)
                    st.metric("With Mask", mask_count)
                    st.metric("Without Mask", no_mask_count)
                    if total_frames > 0:
                        compliance = (mask_count / total_frames) * 100
                        st.metric("Compliance Rate", f"{compliance:.1f}%")
                
                time.sleep(0.03)  # ~30 FPS
            
            cap.release()
    
    else:  # Upload Image mode
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to detect face masks"
        )
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            processed_image, mask_det, no_mask_det, face_count = process_frame(image)
            processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            st.image(processed_image_rgb, channels="RGB", use_container_width=True)
            
            # Display results
            with stats_placeholder.container():
                st.metric("Faces Detected", face_count)
                if mask_det:
                    st.success("✅ Mask Detected!")
                if no_mask_det:
                    st.error("❌ No Mask Detected!")

with col2:
    st.markdown("### 🎯 Detection Status")
    status_placeholder = st.empty()
    
    # Real-time status indicator
    if detection_mode == "Webcam" and run_webcam:
        status_placeholder.markdown("""
            <div class="status-card active">
                <h3>🟢 Active</h3>
                <p>Camera is running</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        status_placeholder.markdown("""
            <div class="status-card inactive">
                <h3>⚪ Inactive</h3>
                <p>Camera is off</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    <div class="tips-card">
        <ul>
            <li>Ensure good lighting</li>
            <li>Face the camera directly</li>
            <li>Keep camera steady</li>
            <li>Avoid multiple people in frame for best results</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        <p>Developed with ❤️ using Streamlit | Powered by VGG16 Deep Learning Model</p>
    </div>
""", unsafe_allow_html=True)