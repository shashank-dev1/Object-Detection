from ultralytics import YOLO
import cv2

# yolov8n,m,l size of the model
model = YOLO('yolov8l.pt')
results = model('crowd.jpg',show=True)
cv2.waitKey(0)
