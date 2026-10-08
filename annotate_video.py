import cv2
from ultralytics import YOLO
import os

VIDEO_PATH = "video/myvideo.mp4"
OUTPUT_PATH = "output/annotated_video.mp4"

os.makedirs("output", exist_ok=True)

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video opened successfully!")
print("Width:", width)
print("Height:", height)
print("FPS:", fps)
print("Total frames:", total_frames)

# Create output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Track only people
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        verbose=False
    )

    annotated_frame = frame.copy()

    for result in results:

        if result.boxes is None:
            continue

        boxes = result.boxes

        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw person rectangle
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # Get tracking ID
            if box.id is not None:
                track_id = int(box.id[0])
                label = f"person ID: {track_id}"
            else:
                label = "person"

            cv2.putText(
                annotated_frame,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    # Show frame number
    cv2.putText(
        annotated_frame,
        f"Frame: {frame_number}/{total_frames}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Person Video Annotation", annotated_frame)

    # Save frame to output video
    out.write(annotated_frame)

    frame_number += 1

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print()
print("Annotation completed!")
print("Annotated video saved at:")
print(OUTPUT_PATH)