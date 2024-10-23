import streamlit as st
import cv2
import tempfile
import os
from pathlib import Path
from tracker import OptimizedOpticalFlowTracker
import time
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Object Tracking App",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton > button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton > button:hover {
        background-color: #FF6B6B;
    }
    .video-container {
        border: 2px solid #f0f0f0;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        background-color: #fafafa;
    }
    .recording-indicator {
        color: red;
        font-weight: bold;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        50% { opacity: 0.5; }
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state variables
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = None
if 'input_video' not in st.session_state:
    st.session_state.input_video = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'camera_video' not in st.session_state:
    st.session_state.camera_video = None
if 'frames_buffer' not in st.session_state:
    st.session_state.frames_buffer = []

def ensure_directory_exists(path):
    """Ensure the directory exists for saving files"""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def save_video_from_frames(frames, output_path, fps=20.0):
    """Save video from frames buffer"""
    if not frames:
        return None
    
    try:
        ensure_directory_exists(output_path)
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        logger.info(f"Video saved successfully to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving video: {str(e)}")
        return None

def process_camera_stream():
    """Real-time camera streaming and processing function"""
    try:
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open camera. Please check your camera connection.")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize the tracker
        tracker = OptimizedOpticalFlowTracker()
        
        # Create container and placeholders once, outside the loop
        stream_container = st.container()
        with stream_container:
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            frame_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            status_placeholder = st.empty()
        
        stop_streaming = st.button("Stop Streaming")
        
        # FPS calculation variables
        fps_start_time = time.time()
        fps_counter = 0
        current_fps = 0
        
        # Streaming loop
        while not stop_streaming:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from camera")
                break
            
            # Calculate FPS
            fps_counter += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time > 1:
                current_fps = fps_counter / elapsed_time
                fps_counter = 0
                fps_start_time = time.time()
            
            # Process frame using the tracker
            processed_frame = tracker.process_frame(frame, current_fps)
            
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Update frame in the existing placeholder
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Update status with FPS
            status_text = "🎥 Live Processing..."
            if current_fps > 0:
                status_text += f" FPS: {current_fps:.2f}"
            status_placeholder.markdown(
                f'<p class="recording-indicator">{status_text}</p>',
                unsafe_allow_html=True
            )
            
            # Add a small delay to prevent overwhelming the UI
            time.sleep(0.01)
        
        # Cleanup
        cap.release()
        frame_placeholder.empty()
        status_placeholder.empty()
        
    except Exception as e:
        logger.error(f"Error during streaming: {str(e)}")
        st.error(f"An error occurred during streaming: {str(e)}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()

def process_video(input_path):
    """Process the video using the OptimizedOpticalFlowTracker"""
    if not os.path.exists(input_path):
        st.error("Input video file not found")
        return None
        
    output_path = os.path.join(tempfile.gettempdir(), "processed_output.mp4")
    tracker = OptimizedOpticalFlowTracker()
    
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error("Could not open input video file")
            return None
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (640, 640))
        
        frame_count = 0
        start_time = time.time()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            progress = frame_count / total_frames if total_frames > 0 else 0
            progress_bar.progress(progress)
            
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            processed_frame = tracker.process_frame(frame, current_fps)
            out.write(processed_frame)
            
            frame_count += 1
            status_text.text(f"Processing frame {frame_count}... FPS: {current_fps:.2f}")
        
        cap.release()
        out.release()
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        return output_path
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        st.error(f"An error occurred during processing: {str(e)}")
        return None

# Main app layout
st.title("🎯 Object Tracking Application")
st.markdown("### Track objects in your videos with advanced AI")

# Create two columns for upload and camera options
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📤 Upload Video")
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            st.session_state.input_video = tfile.name
            
            st.markdown("#### Original Video")
            with st.container():
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                st.video(uploaded_file)
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")

with col2:
    st.markdown("### 📹 Live Camera Processing")
    if st.button("Start Live Processing", key="start_streaming"):
        process_camera_stream()

# Process button
if st.session_state.input_video and os.path.exists(st.session_state.input_video):
    if st.button("🚀 Process Video"):
        with st.spinner("Processing video..."):
            output_path = process_video(st.session_state.input_video)
            if output_path and os.path.exists(output_path):
                st.session_state.processed_video = output_path
                st.session_state.processing_complete = True

# Display processed video and download button
if st.session_state.processing_complete and st.session_state.processed_video:
    st.markdown("### 🎥 Processed Video")
    with st.container():
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.video(st.session_state.processed_video)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Download button
    try:
        with open(st.session_state.processed_video, 'rb') as f:
            st.download_button(
                label="⬇️ Download Processed Video",
                data=f.read(),
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
    except Exception as e:
        st.error(f"Error preparing download: {str(e)}")

# Footer
st.markdown("---")
st.markdown("### 📝 Instructions")
st.markdown("""
1. Choose between uploading a video file or recording from your camera
2. If recording from camera, click 'Start Recording' and 'Stop Recording' when done
3. For either option, click the 'Process Video' button to start object tracking
4. Wait for processing to complete - you'll see a progress bar
5. View the processed video and download it if desired
""")

# Cleanup function
def cleanup():
    """Clean up temporary files"""
    try:
        if st.session_state.input_video and os.path.exists(st.session_state.input_video):
            os.unlink(st.session_state.input_video)
        if st.session_state.processed_video and os.path.exists(st.session_state.processed_video):
            os.unlink(st.session_state.processed_video)
        if st.session_state.camera_video and os.path.exists(st.session_state.camera_video):
            os.unlink(st.session_state.camera_video)
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

# Register cleanup function
import atexit
atexit.register(cleanup)