import cv2
import numpy as np
import argparse

# Создание парсера аргументов командной строки
parser = argparse.ArgumentParser(description="Ball tracking using Kalman filter")
parser.add_argument(
    "--input", type=str, help="path to input video file or camera index", default=0
)
parser.add_argument("--z", type=str, help="height of table", default=50)
args = parser.parse_args()

# Загрузка видеофайла или камеры
cap = cv2.VideoCapture(args.input)

# Настройка цветового фильтра для обнаружения желтого шара
yellow_lower = (22, 93, 0)
yellow_upper = (50, 255, 255)

# Создание объекта фильтра Кальмана
dt = 1 / 30.0
kalman_filter = cv2.KalmanFilter(4, 2)
kalman_filter.measurementMatrix = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
)
kalman_filter.transitionMatrix = np.array(
    [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
)
kalman_filter.processNoiseCov = (
    np.array(
        [[1e-3, 0, 0, 0], [0, 1e-3, 0, 0], [0, 0, 1e-3, 0], [0, 0, 0, 1e-3]],
        dtype=np.float32,
    )
    * 10
)

# Обработка каждого кадра
while True:
    ret, frame = cap.read()

    # Проверка, что кадр был успешно получен
    if not ret:
        break

    # Применение цветового фильтра для обнаружения желтого шара
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Нахождение контуров шара
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Обработка наибольшего контура
    if contours:
        biggest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest_contour)

        # Получение измерений (координаты центра)
        measurements = np.array([[np.float32(x + w / 2)], [np.float32(y + h / 2)]])

        # Обновление фильтра Кальмана
        kalman_filter.correct(measurements)
        prediction = kalman_filter.predict()

        # Отображение ограничивающего прямоугольника
        predicted_x, predicted_y = prediction[0], prediction[1]
        print(f"X: {predicted_x[0]}, Y: {predicted_y[0]}, Z: {args.z}")
        cv2.rectangle(
            frame,
            (int(predicted_x - w / 2), int(predicted_y - h / 2)),
            (int(predicted_x + w / 2), int(predicted_y + h / 2)),
            (255, 0, 0),
            2,
        )

    # Отображение кадра
    cv2.imshow("frame", frame)

    # Прерывание цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Освобождение ресурсов и закрытие окна
cap.release()
cv2.destroyAllWindows()
