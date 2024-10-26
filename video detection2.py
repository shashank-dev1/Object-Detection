from ultralytics import YOLO
import cv2
import cvzone 
import math 

# Initialize video capture
cap = cv2.VideoCapture('video cars.mp4')

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Consider fine-tuning with a custom dataset

# Class names for detection
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

while True:
    success, img = cap.read()
    if not success:
        break  # Exit the loop if the video ends

    # Resize the image for better performance (optional)
    img_resized = cv2.resize(img, (320, 320))  # Resize to model's expected input size

    # Get predictions from the model
    results = model(img_resized, stream=True)

    for r in results:
        boxes = r.boxes 
        for box in boxes:
            # Bounding Box Coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Width and Height
            w, h = x2 - x1, y2 - y1
            
            # Draw Bounding Box
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100  # Confidence score

            # Class Index
            cls = int(box.cls[0])

            # Display Class Name and Confidence
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(32, y1)), scale=1, thickness=1)

    # Show the image with detections
    cv2.imshow('image', img)
    cv2.waitKey(1)  # Allow real-time display

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
