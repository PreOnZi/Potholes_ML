import os
import cv2
from ultralytics import YOLO

# Define the absolute path to the videos directory
VIDEOS_DIR = '/Users/ondrejzika/Desktop/potholes/YOLO/VIDEOS_DIR'

# Define the absolute path to the input video file
video_path = os.path.join(VIDEOS_DIR, '03_1.mp4')
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
custom_model_path = '/Users/ondrejzika/Desktop/Pothole01/pothole_dataset_v8/runs/detect/yolov8n_v8_50e11/weights/best.pt'

# Load the custom model
model = YOLO(custom_model_path)

threshold = 0.5
pothole_detected = False

while ret:
    results = model(frame)[0]

    pothole_detected_in_frame = False

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            pothole_detected_in_frame = True
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    if pothole_detected_in_frame:
        pothole_detected = True
        print('XXXXX')
    else:
        print('x')

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
