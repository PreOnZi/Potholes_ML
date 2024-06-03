import os
import cv2
from ultralytics import YOLO

# Define the absolute path to the videos directory
VIDEOS_DIR = '/Users/ondrejzika/Desktop/Pothole01/pothole_dataset_v8/outputs'

# Define the absolute path to the input video file
video_path = os.path.join(VIDEOS_DIR, 'TestPlot.mp4')
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

while ret:
    results = model(frame)

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Extract bounding box coordinates
                score = box.conf.item()  # Confidence score
                class_id = int(box.cls.item())  # Convert class_id tensor to int
                class_name = model.names[class_id].strip().lower()  # Retrieve the class name for the current detection

                if score > confidence_threshold:  # Apply confidence threshold
                    if class_name == pothole_class_name:
                        print("PPP")

                    # Draw rectangle and put text for visualization (optional)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)  # Green color
                    text = f"{class_name}: {score:.2f}"
                    cv2.putText(frame, text, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Video', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
