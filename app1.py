import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64

# Threshold for object detection
default_threshold = 0.4
image_size = 320

# Convert Hex to BGR (OpenCV format)
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    return (rgb[2], rgb[1], rgb[0])  # OpenCV uses BGR instead of RGB

# Define YOLO function to get bounding boxes
def best_bounding_boxes(output_nn, threshold):
    bounding_box_locations = []
    class_ids = []
    confidence_ = []
    for j in output_nn:
        for k in j:
            class_prob = k[5:]
            class_prob_index = np.argmax(class_prob)
            confidence = class_prob[class_prob_index]
            if confidence > threshold:
                w, h = int(k[2] * image_size), int(k[3] * image_size)
                # Finding x and y
                x, y = int(k[0] * image_size - w / 2), int(k[1] * image_size - h / 2)
                bounding_box_locations.append([x, y, w, h])
                class_ids.append(class_prob_index)
                confidence_.append(float(confidence))
    final_bounding_box = cv2.dnn.NMSBoxes(bounding_box_locations, confidence_, threshold, 0.5)
    return final_bounding_box, bounding_box_locations, confidence_, class_ids

def final_detection(final_box, all_box, acc, classes_index, height_ratio, width_ratio, car_image, total_class_names, show_confidence, box_color, text_color):
    for p in final_box:
        x, y, w, h = all_box[p]
        x = int(x * width_ratio)
        y = int(y * height_ratio)
        w = int(w * width_ratio)
        h = int(h * width_ratio)

        label = str(total_class_names[classes_index[p]])
        conf_ = str(round(acc[p], 2)) if show_confidence else ""
        font = cv2.FONT_HERSHEY_PLAIN
        # Use BGR colors for OpenCV
        cv2.putText(car_image, label + ' ' + conf_, (x, y - 2), font, 2, text_color, 2)
        cv2.rectangle(car_image, (x, y), (x + w, y + h), box_color, 2)

# Load YOLO network and class labels
yolo_neural_network = cv2.dnn.readNetFromDarknet('yolov3.cfg.txt', 'yolov3.weights')
with open('class_names.txt', 'r') as f:
    total_class_names = [line.strip() for line in f.readlines()]

# Streamlit app begins
st.set_page_config(page_title="YOLO Object Detection", page_icon="🔍", layout="wide")

# Custom CSS for Advanced Styling
st.markdown("""
    <style>
        .stApp {
            background-color: #f1f1f1;
            font-family: 'Roboto', sans-serif;
        }
        .title {
            font-size: 50px;
            font-weight: 700;
            color: white;
            background: linear-gradient(45deg, #ff6b6b, #f5a623, #42a5f5);
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            text-transform: uppercase;
            letter-spacing: 3px;
            animation: pulse 2s infinite ease-in-out;
        }
        .description {
            font-size: 18px;
            color: #34495e;
            text-align: center;
            margin-bottom: 40px;
            font-weight: bold;
        }
        @keyframes pulse {
            0% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.1);
            }
            100% {
                transform: scale(1);
            }
        }
        .upload-section {
            text-align: center;
            margin-bottom: 30px;
        }
        .upload-button {
            background-color: #2980b9;
            color: white;
            padding: 15px 25px;
            border-radius: 5px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            border: none;
            transition: background-color 0.3s ease-in-out;
        }
        .upload-button:hover {
            background-color: #3498db;
        }
        .stDownloadButton>button {
            background-color: #27ae60;
            color: white;
            padding: 12px 25px;
            border-radius: 5px;
            font-size: 18px;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }
        .stDownloadButton>button:hover {
            background-color: #2ecc71;
        }
        .stSlider {
            width: 80%;
            margin: 0 auto;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            font-size: 16px;
            color: #7f8c8d;
        }
    </style>
""", unsafe_allow_html=True)

# Hero Section - Colorful Title
st.markdown('<div class="title">YOLO Object Detection with Streamlit 🔍</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Upload an image to detect objects with YOLOv3. Adjust the confidence threshold and visualize the results instantly!</div>', unsafe_allow_html=True)

# Display the YOLO image (centered) using st.image()
yolo_image_path = "yolo_image.jpg"  # Replace this with the correct path to your image file
try:
    st.image(yolo_image_path, caption="YOLOv3 Object Detection Image", use_column_width=True)
except FileNotFoundError:
    st.error("Image file not found. Please make sure yolo_image.jpg is in the same directory.")

# Upload Image Section
st.markdown('<div class="upload-section"><button class="upload-button">Click to Upload an Image</button></div>', unsafe_allow_html=True)

# Upload Image
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Show loading spinner while image is processing
    with st.spinner('Processing Image...'):
        # Convert the uploaded file to a PIL Image
        image = Image.open(uploaded_file)
        car_image = np.array(image)

        # Encode the uploaded image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Display Image Preview using base64
        st.markdown(f'<div class="image-container"><img src="data:image/png;base64,{img_str}" width="100%"></div>', unsafe_allow_html=True)

        # Resize image to fit YOLO model input size
        image_width, image_height = image.size
        input_image_blob = cv2.dnn.blobFromImage(car_image, 1/255, (image_size, image_size), True, crop=False)

        # Get the output layer names for YOLO
        total_layer_names = yolo_neural_network.getLayerNames()
        yolo_layer_names = yolo_neural_network.getUnconnectedOutLayersNames()
        yolo_layer_names_index = yolo_neural_network.getUnconnectedOutLayers()
        proper_layer_index = [total_layer_names[i - 1] for i in yolo_layer_names_index]

        # Get the height and width ratio to scale back the bounding box
        height_ratio = image_height / image_size
        width_ratio = image_width / image_size

        # Add sliders for adjusting features
        threshold = st.slider("Detection Confidence Threshold", 0.0, 1.0, default_threshold, 0.01)
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        box_color = st.color_picker("Bounding Box Color", "#FF0000")
        text_color = st.color_picker("Text Color", "#00FF00")

        # Convert selected colors to BGR
        box_color_bgr = hex_to_bgr(box_color)
        text_color_bgr = hex_to_bgr(text_color)

        # Make prediction using YOLO
        yolo_neural_network.setInput(input_image_blob)
        outputs_from_nn = yolo_neural_network.forward(proper_layer_index)

        # Get bounding boxes for detected objects
        final_box, all_box, acc, class_ids = best_bounding_boxes(outputs_from_nn, threshold)

        # Display final detection on image
        final_detection(final_box, all_box, acc, class_ids, height_ratio, width_ratio, car_image, total_class_names, show_confidence, box_color_bgr, text_color_bgr)

        # Convert the image to PIL format and display it in Streamlit
        result_image = Image.fromarray(car_image)
        st.image(result_image, caption="Detected Image", use_column_width=True)

# Footer Section
st.markdown('<div class="footer">Created  ❤️ by Rajasekhar Naidu and Purendeeswareddy | YOLOv3 Object Detection Demo</div>', unsafe_allow_html=True)
