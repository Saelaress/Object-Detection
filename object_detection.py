from ultralytics import YOLO
import cv2
import numpy as np
import os

# Загрузка предварительно обученной модели YOLO
model = YOLO('yolov8l.pt')

video_path = input("Введите путь к видеофайлу: ")

# Проверка существования указанного файла
if not os.path.exists(video_path):
    print("Указанный файл не найден.")
    exit()

class_name = input("Введите имя класса объекта: ")

# Получение номера класса
try:
    class_number = next(key for key, value in model.names.items() if value == class_name)
except StopIteration:
    print("Указанный класс не найден.")
    exit()

# Открытие видеофайла
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
        # Запуск трекинга YOLOv8 на кадре
        results = model.track(source=frame, conf=0.65, iou=0.5, classes=class_number)

        if results[0].boxes.cls.numel() > 0:
            # Визуализируем результаты на кадре с указанными параметрами
            annotated_frame = results[0].plot()

            # Извлекаем координаты ограничительных рамок обнаруженных объектов
            bboxes = results[0].boxes.xyxy.cpu().numpy()

            # Добавляем место для текста аннотаций
            bboxes[:, :4] += [0, -50, 50, 0]

            # Убедимся, что координаты остаются в пределах размеров изображения
            bboxes[:, :2] = np.maximum(bboxes[:, :2], 0)  # левый верхний угол
            bboxes[:, 2:] = np.minimum(bboxes[:, 2:], frame.shape[:2][::-1])  # правый нижний угол

            # Создаем новый пустой кадр
            new_frame = np.zeros((frame.shape), dtype='uint8')

            for bbox in bboxes:
                x1, y1, x2, y2 = bbox.astype(int)  # Преобразование координат в целые числа

                # Вырезаем область из аннотированного кадра
                cropped_region = annotated_frame[y1:y2, x1:x2]

                # Обновляем новый кадр вырезанной областью
                new_frame[y1:y2, x1:x2] = cropped_region

            # Записываем новый кадр в выходное видео
            output.write(new_frame)
    else:
        # Прерывание цикла при достижении конца видео
        break

# Освобождение ресурсов
cap.release()
output.release()

print(f"Видео успешно сохранено в {output_path}")
