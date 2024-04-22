from ultralytics import YOLO
import cv2

# Загрузка предварительно обученной модели YOLO
model = YOLO('yolov8l.pt')

# Открытие видеофайла
video_path = "resources/1.mp4"
cap = cv2.VideoCapture(video_path)

# Получение ширины и высоты кадра из видео
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Формирование кортежа с размерами кадра для видеовыхода
size = (frame_width, frame_height)

# Указание пути для сохранения результирующего видео и создание объекта для записи видео
output_path = "result.avi"
output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), 20, size)

while cap.isOpened():
    # Чтение кадра из видео
    success, frame = cap.read()
    if success:
        # Запуск трекинга YOLOv8 на кадре, сохранение треков между кадрами
        results = model.track(source=frame, save=True, conf=0.65, iou=0.5, classes=15)
        if results[0].boxes.cls.numel() > 0:
            # Визуализация результатов на кадре
            annotated_frame = results[0].plot()
            output.write(annotated_frame)
    else:
        # Прерывание цикла при достижении конца видео
        break

# Освобождение объекта захвата видео и закрытие окон отображения
cap.release()
output.release()
cv2.destroyAllWindows()

print(f"Видео успешно сохранено в {output_path}")
