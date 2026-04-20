from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import random

# Load model
model = YOLO("yolov8s.pt")  # better accuracy than yolov8n

# Load image
image_path = "kavan1.jpg"
image = cv2.imread(image_path)

# Run detection
results = model(image)

# Dictionary for counting objects
object_counts = {}

# Process results
for r in results:
    for box in r.boxes:
        # Confidence
        conf = float(box.conf[0])
        if conf < 0.5:  # filter low confidence
            continue

        # Class ID and name
        cls_id = int(box.cls[0])
        label = r.names[cls_id]

        # Count objects
        object_counts[label] = object_counts.get(label, 0) + 1

        # Bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Random color
        color = [random.randint(0,255) for _ in range(3)]

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label text
        text = f"{label} {conf:.2f}"
        cv2.putText(image, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 2)

# Show image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Advanced Object Detection")
plt.show()

# Save output image
cv2.imwrite("output.jpg", image)

# Print results
print("\n🔍 Detected Objects:")
for obj, count in object_counts.items():    
    print(f"{obj}: {count}")