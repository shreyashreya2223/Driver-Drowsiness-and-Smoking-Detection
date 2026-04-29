import cv2
import numpy as np
import pygame
from ultralytics import YOLO

print("Starting Driver Monitoring System...")

# =====================================
# LOAD YOLOV26n DROWSINESS MODEL
# =====================================

try:
    drowsiness_model = YOLO("../drowsiness_model/drowsiness_model.pt")
    print("Drowsiness YOLO model loaded successfully!")
except Exception as e:
    print(f"Error loading drowsiness model: {e}")
    exit()

# =====================================
# LOAD YOLOV26n SMOKING MODEL
# =====================================

try:
    smoking_model = YOLO("../smoke_detection/smoking_model.pt")
    print("Smoking YOLO model loaded successfully!")
except Exception as e:
    print(f"Error loading smoking model: {e}")
    exit()

# =====================================
# ALARM SETUP
# =====================================

pygame.mixer.init()

try:
    pygame.mixer.music.load("../drowsiness_model/alarm.mp3")
except:
    print("Alarm file not found.")

alarm_playing = False

# =====================================
# CAMERA START
# =====================================

cap = cv2.VideoCapture(0)

print("System Running... Press Q to Quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # =====================================
    # DROWSINESS DETECTION (YOLOv26n)
    # =====================================

    drowsy_detected = False

    drowsy_results = drowsiness_model(frame)

    for result in drowsy_results:
        for box in result.boxes:
            cls = int(box.cls[0])
            class_name = result.names[cls]

            # Trigger for dangerous classes
            if class_name in ["microsleep", "yawning"]:
                drowsy_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Red box for drowsiness
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"DROWSY: {class_name}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

    # =====================================
    # SMOKING DETECTION (YOLOv26n)
    # =====================================

    smoking_detected = False

    smoke_results = smoking_model(frame)

    for result in smoke_results:
        for box in result.boxes:
            cls = int(box.cls[0])
            class_name = result.names[cls]

            # Trigger only smoking class
            if class_name == "smoking":
                smoking_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Blue box for smoking
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "SMOKING DETECTED",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )

    # =====================================
    # STATUS DISPLAY
    # =====================================

    if not drowsy_detected and not smoking_detected:
        cv2.putText(
            frame,
            "ACTIVE",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # =====================================
    # ALARM LOGIC
    # =====================================

    if drowsy_detected or smoking_detected:
        if not alarm_playing:
            pygame.mixer.music.play(-1)
            alarm_playing = True
    else:
        if alarm_playing:
            pygame.mixer.music.stop()
            alarm_playing = False

    # =====================================
    # SHOW WINDOW
    # =====================================

    cv2.imshow(
        "Driver Monitoring System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()