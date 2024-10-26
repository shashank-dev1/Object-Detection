from ultralytics import YOLO
import cv2
import cvzone 
import math 

# Load the image
img = cv2.imread('crowd.jpg')  # Replace 'your_image.jpg' with the path to your image

# Load the YOLO model
model = YOLO('yolov8n.pt')

# List of class names
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", 
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", 
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", 
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", 
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", 
    "sports ball", "kite", "baseball glove", "skateboard", "surfboard", 
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", 
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", 
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", 
    "pottedplant", "bed", "diningtable", "toilet", "TV monitor", "laptop", 
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
    "toaster", "sink", "refrigerator", "book", "clock", "vase", 
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Perform object detection
results = model(img, stream=True)
for r in results:
    boxes = r.boxes 
    for box in boxes:
        # Bounding Box
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(img, (x1, y1, w, h))

        # Confidence
        conf = math.ceil((box.conf[0] * 100)) / 100

        # Class name
        cls = int(box.cls[0])
        cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(32, y1)), scale=1, thickness=1)
# saving img
        cv2.imwrite('output_image.jpg', img)

# Display the image with detections
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
