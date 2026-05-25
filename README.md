# 🛒Smart-Cart
Smart shopping cart using custom YOLOv8 training, object tracking, and basket zone detection.
The project detects products placed inside a shopping basket zone and automatically updates the shopping cart and total price in real time.

---


## ⚙️ How It Works

1. The webcam captures live video frames.
2. YOLOv8 detects objects in real time.
3. ByteTrack assigns tracking IDs to detected objects.
4. The system checks whether an object overlaps with the basket zone.
5. If the object stays inside the basket zone for several frames, it is added to the shopping cart.
6. If the object leaves the basket zone for several frames, it is removed from the cart.
7. The system updates the total price automatically.


# Demo
<img src="https://github.com/mina91gh/Smart-Cart/blob/main/Smart-Cart.mp4" width="500">



# Model Training

The object detection model was trained using a custom dataset and YOLOv8.

The training process included:

- Collecting and labeling custom product images
- Creating annotations using Roboflow
- Organizing the dataset into train/validation sets
- Training a YOLOv8s model using Ultralytics
- Evaluating detection performance
- Exporting the best trained weights (`best.pt`)

---

## Dataset

The dataset contains labeled images of shopping-related objects such as:

- Remote
- Cell-Phone
- Glasses

Different lighting conditions, object positions, rotations, and distances were included to improve model robustness and real-time detection performance.

---

## Annotation

Dataset annotation was performed using Roboflow.

Bounding boxes were created for each object class to train the detection model.

The dataset was exported in YOLOv8 format.

---

## Training Configuration

Training was performed using the Ultralytics YOLOv8 framework.

Example training command:

```bash
yolo detect train \
data=data.yaml \
model=yolov8s.pt \
epochs=70 \
imgsz=640 \
batch=8
```

---

## Model Output

After training, the best model weights were saved as:

```text
runs/detect/train/weights/best.pt
```

The trained model is then used for real-time detection and tracking inside the Smart Cart system.

---

## Tracking System

The project uses ByteTrack for multi-object tracking.

Tracking IDs are assigned to detected objects to maintain object consistency across video frames.

Additional custom logic was implemented to reduce ID switching by comparing object positions and labels between frames.

---

## 🛒Basket Zone Logic

A custom overlap-based basket detection system was implemented.

The system calculates the overlap ratio between:
- detected object bounding boxes
- basket zone region

Objects are only added to the cart after remaining inside the basket zone for multiple consecutive frames, improving stability and reducing false positives.

Similarly, objects are removed only after staying outside the basket zone for several frames.

---





