from ultralytics import YOLO

# Загрузка предварительно обученной модели YOLO
model = YOLO('yolov8l.pt')

results = model.track(source="resources/1.mp4", save=True, conf=0.65, iou=0.5, classes=15)
