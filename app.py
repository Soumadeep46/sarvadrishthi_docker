import streamlit as st
from datetime import datetime
import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
import requests
from dotenv import load_dotenv
import io

# NEW: imports for WebRTC
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase

load_dotenv()

st.set_page_config(page_title="Facial Recognition System", layout="wide")

st.markdown("""
    <style>
        :root {
            --primary-color: #2563eb;
            --secondary-color: #1e40af;
            --text-color: #1f2937;
            --light-bg: #f3f4f6;
        }
        body {
            font-family: system-ui, -apple-system, sans-serif;
            background-color: var(--light-bg);
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stMarkdown h1 {
            color: var(--primary-color);
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .stMarkdown h2 {
            color: var(--secondary-color);
            text-align: center;
        }
        .stSidebar .stMarkdown {
            color: white;
        }
        .stSidebar {
            background-color: var(--primary-color);
            color: white;
        }
        .footer {
            background-color: var(--text-color);
            color: white;
            text-align: center;
            padding: 1rem;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Pinata API setup (unchanged)
PINATA_JWT_TOKEN = os.getenv('PINATA_JWT_TOKEN')
PINATA_API_URL = "https://api.pinata.cloud/pinning/pinFileToIPFS"
PINATA_GATEWAY = "https://gateway.pinata.cloud/ipfs/"

ATTENDANCE_FILE = "Attendance.csv"

nav_options = ["Home", "Attendance System", "Features", "About"]
selected_nav = st.sidebar.radio("Navigation", nav_options)

if selected_nav == "Home":
    st.markdown("<h1>Facial Recognition Technology</h1>", unsafe_allow_html=True)
    st.markdown("""
    ### Exploring the Future of Biometric Identification and Security
    
    Our advanced facial recognition system provides:
    - Real-time face detection
    - High-accuracy identification
    - Secure and privacy-focused technology
    """)

elif selected_nav == "Features":
    st.markdown("<h2>Key Features</h2>", unsafe_allow_html=True)
    features = [
        ("Real-time Detection", "Advanced algorithms for instant face detection and recognition in live video streams."),
        ("High Accuracy", "State-of-the-art deep learning models ensuring precise identification and minimal false positives."),
        ("Security First", "Built-in privacy protection and data encryption to maintain user confidentiality.")
    ]
    cols = st.columns(3)
    for i, (title, description) in enumerate(features):
        with cols[i]:
            st.markdown(f"### {title}")
            st.write(description)

elif selected_nav == "Attendance System":
    st.markdown("<h1>SARVADRISHTI</h1>", unsafe_allow_html=True)

    if 'attendance_buffer' not in st.session_state:
        st.session_state.attendance_buffer = []
    if 'marked_names' not in st.session_state:
        st.session_state.marked_names = set()
    if 'last_attendance_names' not in st.session_state:
        st.session_state.last_attendance_names = set()

    current_date = st.sidebar.date_input("Date", datetime.today())
    current_time = st.sidebar.time_input("Time", datetime.now().time())
    stop_button = st.sidebar.button("Stop Attendance System")

    # --------- IPFS helpers (unchanged) ----------
    def fetch_from_pinata(cid):
        response = requests.get(f"{PINATA_GATEWAY}{cid}")
        response.raise_for_status()
        return response.content

    def upload_to_pinata(file_content, file_name):
        headers = {'Authorization': f'Bearer {PINATA_JWT_TOKEN}'}
        files = {'file': (file_name, file_content)}
        response = requests.post(PINATA_API_URL, headers=headers, files=files)
        response.raise_for_status()
        return response.json()

    uploaded_image = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        try:
            pinata_response = upload_to_pinata(uploaded_image.getvalue(), uploaded_image.name)
            if 'IpfsHash' in pinata_response:
                st.sidebar.success("Image uploaded successfully to IPFS!")
                st.sidebar.write(f"IPFS Hash: {pinata_response['IpfsHash']}")
            else:
                st.sidebar.error("Failed to upload image to IPFS.")
        except Exception as e:
            st.sidebar.error(f"IPFS error: {e}")

    @st.cache_resource
    def load_images_and_encode():
        images, classNames = [], []
        response = requests.get("https://api.pinata.cloud/data/pinList",
                                headers={'Authorization': f'Bearer {PINATA_JWT_TOKEN}'})
        response.raise_for_status()
        pinata_list = response.json()
        for item in pinata_list.get('rows', []):
            if item['metadata']['name'].lower().endswith(('.png', '.jpg', '.jpeg')):
                img_content = fetch_from_pinata(item['ipfs_pin_hash'])
                nparr = np.frombuffer(img_content, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    images.append(img)
                    classNames.append(os.path.splitext(item['metadata']['name'])[0])
        return images, classNames

    @st.cache_resource
    def find_encodings(images):
        encodings = []
        for idx, img in enumerate(images):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                encoding = face_recognition.face_encodings(img_rgb)[0]
                encodings.append(encoding)
            except IndexError:
                st.warning(f"No face detected in image {idx + 1}. Skipping...")
        return encodings

    images, classNames = load_images_and_encode()
    encodeListKnown = find_encodings(images)
    st.sidebar.success(f"Loaded {len(encodeListKnown)} face encodings from IPFS.")

    def load_attendance():
        if not os.path.exists(ATTENDANCE_FILE) or os.stat(ATTENDANCE_FILE).st_size == 0:
            attendance_df = pd.DataFrame(columns=["Name", "Date", "Time"])
            attendance_df.to_csv(ATTENDANCE_FILE, index=False)
            return attendance_df
        return pd.read_csv(ATTENDANCE_FILE)

    attendance_df = load_attendance()
    st.sidebar.markdown("### Attendance History")
    st.sidebar.dataframe(attendance_df, use_container_width=True)

    def save_attendance_buffer():
        if st.session_state.attendance_buffer:
            attendance_df = pd.read_csv(ATTENDANCE_FILE)
            new_entries = pd.DataFrame(st.session_state.attendance_buffer)
            attendance_df = pd.concat([attendance_df, new_entries], ignore_index=True)
            attendance_df.drop_duplicates(subset=["Name", "Date"], keep="last", inplace=True)
            attendance_df.to_csv(ATTENDANCE_FILE, index=False)
            st.session_state.attendance_buffer = []

    st.markdown("<h2>Tracking System</h2>", unsafe_allow_html=True)

    # ---------- WebRTC camera (replaces cv2.VideoCapture loop) ----------
    # Use public STUN so ICE works on Cloud/Docker
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    # Face recognition processor using streamlit-webrtc
    class AttendanceProcessor(VideoProcessorBase):
        def __init__(self):
            self.encodeListKnown = encodeListKnown
            self.classNames = classNames
            self.tolerance = 0.4
            self.min_conf = 0.60
            self.caught_names = set()

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Downscale like before
            imgS = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(imgS, model="hog")
            encodings = face_recognition.face_encodings(imgS, face_locations)

            for encoding, face_location in zip(encodings, face_locations):
                matches = face_recognition.compare_faces(self.encodeListKnown, encoding, tolerance=self.tolerance)
                face_distances = face_recognition.face_distance(self.encodeListKnown, encoding)

                if matches and len(face_distances) > 0:
                    match_index = int(np.argmin(face_distances))
                    confidence = float(1 - face_distances[match_index])

                    if confidence > self.min_conf:
                        name = self.classNames[match_index]
                        self.caught_names.add(name)

                        y1, x2, y2, x1 = [v * 2 for v in face_location]
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, f"{name} ({confidence:.2f})", (x1, y2 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        y1, x2, y2, x1 = [v * 2 for v in face_location]
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(img, "Unknown", (x1, y2 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            return frame.from_ndarray(img, format="bgr24")

    ctx = webrtc_streamer(
        key="sarvadrishti-webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=AttendanceProcessor,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Attendance Now"):
            if ctx and ctx.video_processor:
                names_to_add = ctx.video_processor.caught_names - st.session_state.marked_names
                for name in names_to_add:
                    st.session_state.attendance_buffer.append({
                        "Name": name,
                        "Date": datetime.now().strftime("%B %d, %Y"),
                        "Time": datetime.now().strftime("%H:%M:%S")
                    })
                    st.session_state.marked_names.add(name)
                ctx.video_processor.caught_names.clear()
            save_attendance_buffer()
            st.success("Attendance saved.")
    with col2:
        if st.button("Clear Today’s Buffer"):
            st.session_state.attendance_buffer = []
            st.session_state.marked_names = set()
            if ctx and ctx.video_processor:
                ctx.video_processor.caught_names.clear()
            st.info("Cleared unsaved entries.")

elif selected_nav == "About":
    st.markdown("<h2>About Our Facial Recognition System</h2>", unsafe_allow_html=True)
    st.markdown("""
    ### Technology Overview
    Our facial recognition technology is designed for secure and efficient identity verification.

    ### Key Principles
    - Privacy Protection
    - Advanced Machine Learning
    - Continuous Improvement
    
    © 2024 Madhya Pradesh Police
    """)

st.markdown("""
    <div class="footer">
        © 2024 Facial Recognition Technology
    </div>
""", unsafe_allow_html=True)
