import os
import cv2
import numpy as np
import random
from ultralytics import YOLO
from pyaxidraw import axidraw
import threading
import time
import math
from concurrent.futures import ThreadPoolExecutor, TimeoutError

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

# Timer variables
frame_timer = None
frame_timer_lock = threading.Lock()
timer_expired = threading.Event()

# ThreadPoolExecutor for plotter operations
executor = ThreadPoolExecutor(max_workers=1)

# Function to draw a random line and then move to home position
def draw_random_line_and_home():
    with ad_lock:
        x1 = random.uniform(0, 320)  # Adjust this value based on plotter's drawing area
        y1 = random.uniform(0, 320)
        x2 = random.uniform(0, 320)
        y2 = random.uniform(0, 320)
        ad.moveto(x1, y1)
        ad.lineto(x2, y2)
        ad.moveto(0, 0)  # Move back to home position

# Function to move plotter back to home position
def move_to_home():
    with ad_lock:
        ad.moveto(0, 0)

# Function to process detections and execute plotter commands
def process_detections(detections):
    threads = []
    for i, box in enumerate(detections):
        if i >= 3:
            break  # Only process up to 3 potholes
        thread = threading.Thread(target=draw_random_line_and_home)
        thread.start()
        threads.append(thread)
    # Join threads with a timeout to prevent blocking
    for thread in threads:
        thread.join(timeout=5)

# Function to reset the frame timer
def reset_frame_timer():
    global frame_timer
    with frame_timer_lock:
        if frame_timer is not None:
            frame_timer.cancel()
        frame_timer = threading.Timer(15, handle_frame_timeout)
        frame_timer.start()
        timer_expired.clear()

# Function to handle frame timeout
def handle_frame_timeout():
    print("Frame processing timeout! Forcing plotter to home position.")
    timer_expired.set()
    executor.submit(move_to_home)

while ret:
    if timer_expired.is_set():
        reset_frame_timer()
        continue  # Skip the rest of the loop and reset the timer

    results = model(frame)
    pothole_detected = False
    detections = []

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
                        detections.append(box)
                        # Draw rectangle and put text for visualization (optional)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)  # Green color
                        text = f"{class_name}: {score:.2f}"
                        cv2.putText(frame, text, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    if pothole_detected:
        # Draw random red lines on the frame
        for _ in range(10):
            color = (0, 0, 255)  # Red color if pothole detected
            pt1 = (np.random.randint(0, W), np.random.randint(0, H))
            pt2 = (np.random.randint(0, W), np.random.randint(0, H))
            cv2.line(frame, pt1, pt2, color, 2)

        # Display the frame with detections before starting plotter operations
        cv2.imshow('Video', frame)
        out.write(frame)
        cv2.waitKey(1)  # Ensure the frame is shown

        # Submit plotter tasks to the executor with a timeout
        future = executor.submit(process_detections, detections)
        try:
            future.result(timeout=15)  # Wait for at most 15 seconds for the plotter to complete
        except TimeoutError:
            print("Plotter task timeout. Moving on to the next frame.")
    else:
        # Draw random green lines on the frame
        for _ in range(10):
            color = (0, 255, 0)  # Green color if no pothole detected
            pt1 = (np.random.randint(0, W), np.random.randint(0, H))
            pt2 = (np.random.randint(0, W), np.random.randint(0, H))
            cv2.line(frame, pt1, pt2, color, 2)

        # Display the frame without detections
        cv2.imshow('Video', frame)
        out.write(frame)

        # Move to home position if no pothole is detected
        executor.submit(move_to_home)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

    # Reset the frame timer
    reset_frame_timer()

cap.release()
out.release()
cv2.destroyAllWindows()

# Cancel the frame timer
with frame_timer_lock:
    if frame_timer is not None:
        frame_timer.cancel()

# Shutdown the executor
executor.shutdown(wait=False)

# Disconnect from the Axidraw plotter
ad.disconnect()
