import cv2
import argparse
from utils import get_dominant_color
from yolov8 import YOLOv8

parser = argparse.ArgumentParser(description="Ball tracking using Kalman filter")
parser.add_argument(
    "--input", type=str, help="path to input video file or camera index", default=0
)
parser.add_argument("--z", type=str, help="height of table", default=50)
args = parser.parse_args()

# # Initialize video
cap = cv2.VideoCapture(args.input)

start_time = 1
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

model_path = "best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.3, iou_thres=0.5)

# out = cv2.VideoWriter(
#     "output.mp4",
#     cv2.VideoWriter_fourcc("M", "J", "P", "G"),
#     cap.get(cv2.CAP_PROP_FPS),
#     (848, 464),
# )
while cap.isOpened():
    # Press key q to stop
    if cv2.waitKey(1) == ord("q"):
        break

    try:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        boxes, scores, class_ids = yolov8_detector(frame)
        for i, box in enumerate(boxes):
            x, y, x2, y2 = list(map(int, box))
            object_image = frame[y:y2, x:x2]
            color = get_dominant_color(object_image)
            if color == 2:
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 1, 1)
                print(f"X: {x}, Y: {y}, Z: {args.z}")
        cv2.imshow("w", frame)
    except Exception as e:
        print(e)
        continue
    # out.write(frame)
