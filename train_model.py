from ultralytics import YOLO

# load base model
model = YOLO("yolov8s.pt")

# train on your dataset
model.train(
    data="data.yaml",
    epochs=70,
    imgsz=640,
    batch=8
)