import os
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Ensure sort.py is in your project directory

# Define the absolute path to the videos directory
VIDEOS_DIR = '/Users/ondrejzika/Desktop/Pothole01/pothole_dataset_v8/outputs'

# Define the absolute path to the input video file
video_path = os.path.join(VIDEOS_DIR, 'RoadTest.mp4')
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
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Define the path to your custom model weights file
custom_model_path = '/Users/ondrejzika/Desktop/Pothole01/pothole_dataset_v8/runs/detect/yolov8n_v8_50e21/weights/best.pt'

# Load the custom model
model = YOLO(custom_model_path)

# Define confidence threshold
confidence_threshold = 0.7  # Adjust this threshold as needed

# Initialize the SORT tracker
tracker = Sort()

# Initialize variable to store the unique tracked potholes
tracked_pothole_ids = set()

while ret:
    results = model(frame)

    detections = []

    for result in results:
        boxes = result.boxes.data.tolist()

        for box in boxes:
            x1, y1, x2, y2, score, class_id = box
            score = float(score)

            # Check if score is above the confidence threshold
            if score > confidence_threshold:
                detections.append([x1, y1, x2, y2, score])

    if len(detections) > 0:
        detections = np.array(detections)
        tracked_objects = tracker.update(detections)

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj[:5])

            # If the object ID is new, add it to the set
            if obj_id not in tracked_pothole_ids:
                tracked_pothole_ids.add(obj_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)  # Green color
            text = f"Pothole ID: {obj_id}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()

# Print the final unique pothole count
print(f"Total unique potholes detected with confidence over {confidence_threshold}: {len(tracked_pothole_ids)}")
