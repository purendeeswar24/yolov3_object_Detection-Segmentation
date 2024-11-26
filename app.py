import cv2
import numpy as np
import streamlit as st
from PIL import Image
import time

# Load YOLO model and class labels
def load_yolo_model(cfg_path, weights_path):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    layer_names = net.getLayerNames()
    output_layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layer_names

# Get bounding boxes from YOLO outputs
def get_bounding_boxes(output_nn, threshold=0.4, image_size=320):
    bounding_boxes = []
    confidences = []
    class_ids = []
    for output in output_nn:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > threshold:
                w, h = int(detection[2] * image_size), int(detection[3] * image_size)
                x, y = int(detection[0] * image_size - w / 2), int(detection[1] * image_size - h / 2)
                bounding_boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return bounding_boxes, confidences, class_ids

# Perform object detection in real-time
def detect_objects(frame, net, output_layer_names, class_names, image_size=320):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (image_size, image_size), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layer_names)
    
    bounding_boxes, confidences, class_ids = get_bounding_boxes(outputs)

    # Non-maximum Suppression (NMS) to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(bounding_boxes, confidences, score_threshold=0.4, nms_threshold=0.5)
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = bounding_boxes[i]
            label = class_names[class_ids[i]]
            confidence = str(round(confidences[i], 2))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} {confidence}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Add CSS styling to the app
def add_css():
    st.markdown(
        """
        <style>
            /* Change background color */
            .stApp {
                background-color: #f4f4f9;
            }
            
            /* Customize title */
            h1 {
                font-family: 'Arial', sans-serif;
                color: #4CAF50;
                text-align: center;
                font-size: 36px;
            }
            
            /* Customizing Streamlit buttons */
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                border-radius: 10px;
                padding: 10px;
                box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.1);
            }
            
            .stButton>button:hover {
                background-color: #45a049;
            }
            
            /* Enhance the dropdown menu */
            .stSelectbox>div>div>input {
                padding: 8px;
                font-size: 16px;
            }
            
            /* Video and image display styling */
            .stImage>img {
                border-radius: 10px;
                box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.15);
            }
            
            /* Customizing file uploader */
            .stFileUploader>div>div>input {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }

            .stFileUploader>div>div>input:hover {
                background-color: #45a049;
            }
            
            /* Customize headers */
            h3 {
                font-family: 'Arial', sans-serif;
                color: #333333;
                text-align: center;
                font-size: 28px;
            }
        </style>
        """, unsafe_allow_html=True
    )

# Show progress bar during processing
def show_progress_bar():
    st.sidebar.subheader("Processing...")
    progress = st.sidebar.progress(0)
    for i in range(100):
        time.sleep(0.05)
        progress.progress(i + 1)

def main():
    # Add custom CSS to the app
    add_css()

    st.title("Real-Time YOLO Object Detection")
    
    # Display an image at the top (obj.webp)
    st.image("obj.webp", caption="Welcome to YOLO Object Detection App", use_container_width=True)

    # Sidebar content for better user experience
    st.sidebar.header("About the App")
    st.sidebar.text("This app uses YOLO for real-time object detection in images, videos, and live webcam feed.")
    st.sidebar.text("Upload images or videos and detect objects in real time.")

    # Load YOLO model
    yolo_cfg_path = 'C:\\Raja\\Data Science\\DEEP LEARNING\\yolov3_video\\yolov3.cfg.txt'
    yolo_weights_path = 'C:\\Raja\\Data Science\\DEEP LEARNING\\yolov3_video\\yolov3.weights'
    net, output_layer_names = load_yolo_model(yolo_cfg_path, yolo_weights_path)

    # Load class labels
    with open('class_names.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    # Upload options for image and video
    upload_option = st.selectbox("Choose upload type", ["Live Video", "Image", "Video"])

    # Show loading spinner if needed
    if upload_option in ["Image", "Video"]:
        show_progress_bar()

    # Live video (Webcam feed)
    if upload_option == "Live Video":
        st.header("Live Webcam Stream")
        
        # Start video capture using OpenCV
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.write("Error: Could not open video stream.")
            return
        
        stframe = st.empty()  # Placeholder for video stream in Streamlit

        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture frame.")
                break

            # Detect objects in the frame
            frame_resized = cv2.resize(frame, (320, 320))  # Resize to 320x320
            frame_detected = detect_objects(frame_resized, net, output_layer_names, class_names)

            # Convert frame to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame_detected, cv2.COLOR_BGR2RGB)

            # Display the frame with bounding boxes
            stframe.image(frame_rgb, channels="RGB", caption="Real-Time Object Detection", use_container_width=True)

            # Add small delay to smooth out the video stream
            time.sleep(0.05)

        # Release the video capture object
        cap.release()

    # Image upload
    elif upload_option == "Image":
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_image is not None:
            # Load and process the image
            img = Image.open(uploaded_image)
            img = np.array(img)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Detect objects in the uploaded image
            img_detected = detect_objects(img_bgr, net, output_layer_names, class_names)

            # Display the detected image
            img_rgb = cv2.cvtColor(img_detected, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Detected Image", use_container_width=True)

    # Video upload
    elif upload_option == "Video":
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi"])

        if uploaded_video is not None:
            video_file = uploaded_video.read()
            st.video(video_file)

            # Additional video processing can go here...

if __name__ == "__main__":
    main()
