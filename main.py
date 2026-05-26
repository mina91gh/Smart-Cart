import cv2
import json

from ultralytics import YOLO
from collections import defaultdict
from math import sqrt

# ============================================
# LOAD PRICES
# ============================================

with open("prices.json", "r") as f:
    prices = json.load(f)

# ============================================
# LOAD MODEL
# ============================================


model = YOLO("best.pt")


# ============================================
# CAMERA
# ============================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ============================================
# BASKET ZONE
# ============================================

ZONE = (180, 150, 650, 700)

# allowed coco classes
allowed_items = ["Remote", "Cell-Phone", "Glasses"]

# ============================================
# SHOPPING CART
# ============================================

cart = defaultdict(int)

# tracking states
track_state = {}

# frame counters
inside_counter = defaultdict(int)
outside_counter = defaultdict(int)

# prevent duplicate counting
already_counted = set()

# object centers memory
last_positions = {}

# stable detection settings
MIN_INSIDE_FRAMES = 5
MIN_OUTSIDE_FRAMES = 12

# re-identification distance
MAX_DISTANCE = 80

# ============================================
# TOTAL PRICE
# ============================================

def get_total():

    total = 0

    for item, count in cart.items():
        total += prices.get(item, 0) * count

    return total

# ============================================
# OVERLAP FUNCTION
# ============================================

def overlap_ratio(box, zone):

    x1, y1, x2, y2 = box
    zx1, zy1, zx2, zy2 = zone

    inter_x1 = max(x1, zx1)
    inter_y1 = max(y1, zy1)

    inter_x2 = min(x2, zx2)
    inter_y2 = min(y2, zy2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0

    intersection = (
        (inter_x2 - inter_x1)
        * (inter_y2 - inter_y1)
    )

    box_area = (x2 - x1) * (y2 - y1)

    return intersection / box_area

# ============================================
# DISTANCE FUNCTION
# ============================================

def distance(p1, p2):

    return sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )

# ============================================
# FIND SIMILAR TRACK
# Helps reduce ID switching
# ============================================

def find_similar_track(center, label):

    for old_id, data in last_positions.items():

        old_center = data["center"]
        old_label = data["label"]

        if old_label != label:
            continue

        d = distance(center, old_center)

        if d < MAX_DISTANCE:
            return old_id

    return None

# ============================================
# MAIN LOOP
# ============================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ========================================
    # YOLO TRACKING
    # ========================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.85,
        iou=0.5,
        verbose=False
    )

    # ========================================
    # PROCESS DETECTIONS
    # ========================================

    for r in results:

        boxes = r.boxes

        if boxes.id is None:
            continue

        for box, track_id, cls in zip(
            boxes.xyxy,
            boxes.id,
            boxes.cls
        ):

            x1, y1, x2, y2 = map(int, box)

            track_id = int(track_id)
            cls = int(cls)

            label = model.names[cls]

            # filter unwanted objects
            if label not in allowed_items:
                continue

            # center point
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            center = (cx, cy)

            # ====================================
            # SIMPLE RE-ID
            # ====================================

            similar_id = find_similar_track(center, label)

            if similar_id is not None:
                track_id = similar_id

            # save latest position
            last_positions[track_id] = {
                "center": center,
                "label": label
            }

            # ====================================
            # OVERLAP CHECK
            # ====================================

            ratio = overlap_ratio(
                (x1, y1, x2, y2),
                ZONE
            )

            now_inside = ratio > 0.20

            # ====================================
            # INITIALIZE TRACK
            # ====================================

            if track_id not in track_state:
                track_state[track_id] = now_inside

            # ====================================
            # INSIDE LOGIC
            # ====================================

            if now_inside:

                inside_counter[track_id] += 1
                outside_counter[track_id] = 0

                # stable add
                if (
                    inside_counter[track_id]
                    >= MIN_INSIDE_FRAMES
                    and track_id not in already_counted
                ):

                    cart[label] += 1

                    already_counted.add(track_id)

                    print(f"{label} ADDED")

            # ====================================
            # OUTSIDE LOGIC
            # ====================================

            else:

                outside_counter[track_id] += 1
                inside_counter[track_id] = 0

                # stable remove
                if (
                    outside_counter[track_id]
                    >= MIN_OUTSIDE_FRAMES
                    and track_id in already_counted
                ):

                    if cart[label] > 0:
                        cart[label] -= 1

                    already_counted.remove(track_id)

                    print(f"{label} REMOVED")

            # update state
            track_state[track_id] = now_inside

            # ====================================
            # COLORS
            # ====================================

            color = (
                (0, 255, 0)
                if now_inside
                else (255, 0, 0)
            )

            # ====================================
            # DRAW BOX
            # ====================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # center point
            cv2.circle(
                frame,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

            # ====================================
            # LABEL
            # ====================================

            text = f"{label} | ID:{track_id}"

            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # overlap %
            overlap_text = f"{int(ratio * 100)}%"

            cv2.putText(
                frame,
                overlap_text,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

    # ============================================
    # DRAW ZONE
    # ============================================

    zx1, zy1, zx2, zy2 = ZONE

    cv2.rectangle(
        frame,
        (zx1, zy1),
        (zx2, zy2),
        (0, 255, 255),
        3
    )

    cv2.putText(
        frame,
        "BASKET ZONE",
        (zx1, zy1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # ============================================
    # DISPLAY CART
    # ============================================

    y = 30

    cv2.putText(
        frame,
        "SHOPPING CART",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    y += 40

    for item, count in cart.items():

        if count > 0:

            price = prices.get(item, 0)

            text = f"{item}: {count} x ${price}"

            cv2.putText(
                frame,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            y += 35


    total_text = f"TOTAL: ${get_total()}"

    cv2.putText(
        frame,
        total_text,
        (20, y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3
    )


    fps = cap.get(cv2.CAP_PROP_FPS)

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (1050, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

  

    cv2.imshow("Smart Trolley PRO", frame)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()