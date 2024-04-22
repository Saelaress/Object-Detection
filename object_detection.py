from ultralytics import YOLO
import cv2
import numpy as np

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
        # Запуск трекинга YOLOv8 на кадре
        results = model.track(source=frame, save=True, conf=0.65, iou=0.5, classes=15)

        if results[0].boxes.cls.numel() > 0:
            # Визуализируем результаты на кадре с указанными параметрами
            annotated_frame = results[0].plot(line_width=4, font_size=4)

            # Извлекаем координаты ограничительных рамок обнаруженных объектов
            bboxes = results[0].boxes.xyxy.cpu().numpy()

            # Добавляем место для текста аннотаций
            bboxes[:, :4] += [0, -33, 0, 0]

            # Создаем новый пустой кадр
            new_frame = np.zeros((frame.shape), dtype='uint8')

            for bbox in bboxes:
                x1, y1, x2, y2 = bbox.astype(int)  # Преобразование координат в целые числа

                # Вырезаем область из аннотированного кадра
                cropped_region = annotated_frame[y1:y2, x1:x2]

                # Заменяем верхнюю половину границы нулями
                cropped_region[:35, 280:] = 0

                # Обновляем новый кадр вырезанной областью
                new_frame[y1:y2, x1:x2] = cropped_region

            # Записываем новый кадр в выходное видео
            output.write(new_frame)
    else:
        # Прерывание цикла при достижении конца видео
        break

# Освобождение объекта захвата видео и закрытие окон отображения
cap.release()
output.release()
cv2.destroyAllWindows()

print(f"Видео успешно сохранено в {output_path}")
