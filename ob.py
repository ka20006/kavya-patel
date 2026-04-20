from ultralytics import YOLO
import cv2
import random

# Load model
model = YOLO("yolov8s.pt")

# Start webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not working")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    object_counts = {}

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            cls_id = int(box.cls[0])
            label = r.names[cls_id]

            # Count objects
            object_counts[label] = object_counts.get(label, 0) + 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Random color
            color = [random.randint(0,255) for _ in range(3)]

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label
            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 2)

    # Show counts on screen
    y_offset = 30
    for obj, count in object_counts.items():
        cv2.putText(frame, f"{obj}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        y_offset += 30

    # Show frame
    cv2.imshow("🔥 Live Object Detection (YOLOv8)", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()