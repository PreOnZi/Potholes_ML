import os
import cv2
import numpy as np
import random
from ultralytics import YOLO
from pyaxidraw import axidraw
import threading
import time

# Initialize AxiDraw
ad = axidraw.AxiDraw()
ad.options.speed_pendown = 50  # Set maximum pen-down speed to 50%
ad.interactive()  # Set AxiDraw to interactive mode
ad.connect()

# Define the absolute path to the videos directory
VIDEOS_DIR = '/Users/ondrejzika/Desktop/Pothole01/pothole_dataset_v8/outputs'

# Define the absolute path to the input video file
video_path = os.path.join(VIDEOS_DIR, 'JAYWICK.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

# Check if the video file exists
if not os.path.isfile(video_path):
    raise FileNotFoundError(f"Video file '{video_path}' not found.")

cap = cv2.VideoCapture(video_path)

# Check if video capture is successful
if not cap.isOpened():
    raise IOError("Error: Cannot open video capture.")

ret, frame = cap.read()

# Check if frame is read successfully
if frame is None:
    raise IOError("Error: Cannot read frame from the video.")

H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Define the path to your custom model weights file
custom_model_path = '/Users/ondrejzika/Desktop/Pothole01/pothole_dataset_v8/runs/detect/yolov8n_v8_50e21/weights/best.pt'

# Check if the model weights file exists
if not os.path.exists(custom_model_path):
    raise FileNotFoundError(f"Model weights file '{custom_model_path}' not found.")

# Load the custom model
model = YOLO(custom_model_path)

# Define the confidence threshold
confidence_threshold = 0.6

# Define the class names we are interested in
pothole_class_name = "pothole".strip().lower()

# Define lock for AxiDraw access
ad_lock = threading.Lock()

# Function to draw a small circle at a specific location
def draw_circle(x, y):
    with ad_lock:
        ad.moveto(x + 1, y)  # Move to the starting point of the circle
        for angle in range(0, 360, 10):  # Draw line segments to approximate the circle
            x_segment = x + math.cos(math.radians(angle))  # Calculate x-coordinate of the segment end point
            y_segment = y + math.sin(math.radians(angle))  # Calculate y-coordinate of the segment end point
            ad.lineto(x_segment, y_segment)

# Function to move plotter back to home position
def move_to_home():
    with ad_lock:
        ad.moveto(0,0)

while ret:
    results = model(frame)
    pothole_detected = False

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Extract bounding box coordinates
                score = box.conf.item()  # Confidence score
                class_id = int(box.cls.item())  # Convert class_id tensor to int
                class_name = model.names[class_id].strip().lower()  # Retrieve the class name for the current detection

                if score > confidence_threshold:  # Apply confidence threshold
                    if class_name == pothole_class_name:
                        pothole_detected = True
                        print("Pothole detected!")
                        # Start a thread to draw the circle at a random x position and y=100
                        random_x = random.randint(0, 250)  # Adjust this value based on plotter's drawing area
                        threading.Thread(target=draw_circle, args=(random_x, 250)).start()

                        # Draw rectangle and put text for visualization (optional)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)  # Green color
                        text = f"{class_name}: {score:.2f}"
                        cv2.putText(frame, text, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    if not pothole_detected:
        move_to_home()  # Move to home position if no pothole is detected

    # Draw random lines on the frame
    for _ in range(10):
        if pothole_detected:
            color = (0, 0, 255)  # Red color if pothole detected
        else:
            color = (0, 255, 0)  # Green color if no pothole detected
        pt1 = (np.random.randint(0, W), np.random.randint(0, H))
        pt2 = (np.random.randint(0, W), np.random.randint(0, H))
        cv2.line(frame, pt1, pt2, color, 2)

    # Display the frame
    cv2.imshow('Video', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()

# Disconnect from the Axidraw plotter
ad.disconnect()
