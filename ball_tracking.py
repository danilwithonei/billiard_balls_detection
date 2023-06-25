import cv2
import argparse
import numpy as np


parser = argparse.ArgumentParser(description="Ball tracking")
parser.add_argument(
    "--input", type=str, help="path to input video file or camera index", default=0
)
parser.add_argument("--z", type=str, help="height of table", default=50)
parser.add_argument("--show", type=bool, help="show window?", default=False)
parser.add_argument("--print_centers", type=bool, help="print results?", default=False)
parser.add_argument("--record", type=bool, help="record?", default=False)
args = parser.parse_args()


def draw_detections(image, boxes, color, label, mask_alpha=0.3):
    """
    Draws bounding boxes and labels on an image.
    """
    mask_img = image.copy()
    det_img = image.copy()

    # Get image dimensions
    img_height, img_width = image.shape[:2]

    # Calculate caption size and text thickness based on image size
    size = min(img_height, img_width) * 0.0009
    text_thickness = int(min(img_height, img_width) * 0.001)

    # Draw bounding boxes and labels on both images
    for box in boxes:
        x, y, w, h = box

        # Draw rectangle on detection image
        cv2.rectangle(det_img, (x, y, w, h), color, 1)

        # Draw filled rectangle on mask image
        cv2.rectangle(mask_img, (x, y, w, h), color, -1)

        # Add label to detection and mask images
        caption = f"{label}"
        cv2.putText(
            det_img,
            caption,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA,
        )

        cv2.putText(
            mask_img,
            caption,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA,
        )

    # Combine mask and detection images with alpha blending
    return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)


def get_balls_bboxes(
    img: np.ndarray,
    lower_hsv: np.ndarray,
    higher_hsv: np.ndarray,
    max_bbox_area: int = 200,
    min_bbox_area: int = 40,
) -> list[tuple[int, int, int, int]]:
    """
    Returns bounding boxes for balls in an image.
    """
    # Blur image and convert to HSV color space
    blur_img = cv2.GaussianBlur(img, (3, 3), 5)
    hsv = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)

    # Create mask based on given lower and higher HSV color values
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

    # Find contours in mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize empty list for storing ball bounding boxes
    bboxes = []

    # Loop over all contours and check if their area is within the specified range
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        c_area = cv2.contourArea(cnt)

        if max_bbox_area > c_area > min_bbox_area:
            bboxes.append((x, y, w, h))

    # Return list of ball bounding boxes
    return bboxes


def get_bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    """
    Returns center point of a bounding box.
    """
    x, y, w, h = bbox
    return (x + int(w / 2), y + int(h / 2))


# Define lower and higher HSV color values for yellow and white balls
y_lower_hsv = np.array([10, 136, 159])
y_higher_hsv = np.array([38, 255, 255])

w_lower_hsv = np.array([19, 10, 222])
w_higher_hsv = np.array([55, 74, 255])

# Open video capture from file
cap = cv2.VideoCapture(args.input)
width = int(cap.get(3))
height = int(cap.get(4))
if args.record:
    out = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc(*"MP4V"),
        cap.get(cv2.CAP_PROP_FPS),
        (width, height),
    )

# Loop over all frames in video
while True:
    ret, img = cap.read()
    if not ret:
        break

    # Get bounding boxes for yellow and white balls in current frame
    yellow_balls_bboxes = get_balls_bboxes(img, y_lower_hsv, y_higher_hsv)
    white_balls_bboxes = get_balls_bboxes(img, w_lower_hsv, w_higher_hsv)

    # Draw bounding boxes and labels on image for each ball type
    img = draw_detections(img, yellow_balls_bboxes, (0, 255, 255), "yellow")
    img = draw_detections(img, white_balls_bboxes, (255, 255, 255), "white")

    if args.print_centers:
        for y_bbox in yellow_balls_bboxes:
            print(
                "yellow bolls center : X:{}, Y:{}, Z:{}".format(
                    *get_bbox_center(y_bbox), args.z
                )
            )
        for w_bbox in white_balls_bboxes:
            print(
                "white bolls center : X:{}, Y:{}, Z:{}".format(
                    *get_bbox_center(w_bbox), args.z
                )
            )

    if args.record:
        out.write(img)  # type:ignore

    # Show image window if flag is set to True
    if args.show:
        cv2.imshow("result", img)

        # Wait for 'q' key to be pressed to exit loop and close window
        if cv2.waitKey(1) == ord("q"):
            break

# Release capture and destroy all windows
cap.release()
out.release()  # type:ignore
cv2.destroyAllWindows()
