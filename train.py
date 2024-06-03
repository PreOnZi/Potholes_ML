from ultralytics import YOLO
model = YOLO ('yolov8n.pt')


results = model.train(
    data='data.yaml',
    imgsz=640,
    epochs=150,
    batch=8,
    name='yolov8n_v8_50e'
)