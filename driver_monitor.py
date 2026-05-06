import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
from datetime import datetime
from ultralytics import YOLO

print("Starting Driver Monitoring System...")
print("Loading AI Models...")
print("Initializing Webcam...")
print("System Ready!")

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
# MEDIAPIPE FACE MESH
# =====================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1
)

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
# LOG FUNCTION
# =====================================

def save_log(message):

    current_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open("../log.txt", "a") as file:

        file.write(
            f"[{current_time}] {message}\n"
        )

# =====================================
# EYE LANDMARKS
# =====================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# =====================================
# EAR FUNCTION
# =====================================

def calculate_EAR(eye):

    A = np.linalg.norm(
        np.array(eye[1]) - np.array(eye[5])
    )

    B = np.linalg.norm(
        np.array(eye[2]) - np.array(eye[4])
    )

    C = np.linalg.norm(
        np.array(eye[0]) - np.array(eye[3])
    )

    ear = (A + B) / (2.0 * C)

    return ear

# =====================================
# DROWSINESS VARIABLES
# =====================================

EAR_THRESHOLD = 0.25
FRAME_LIMIT = 15

closed_frames = 0

# =====================================
# FPS VARIABLES
# =====================================

prev_time = 0

# =====================================
# CAMERA START
# =====================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("System Running... Press Q to Quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # =====================================
    # PROJECT TITLE
    # =====================================

    cv2.putText(
        frame,
        "AI DRIVER MONITORING SYSTEM",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # =====================================
    # DROWSINESS DETECTION
    # =====================================

    drowsy_detected = False

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            h, w, _ = frame.shape

            left_eye = []
            right_eye = []

            # LEFT EYE
            for idx in LEFT_EYE:

                x = int(
                    face_landmarks.landmark[idx].x * w
                )

                y = int(
                    face_landmarks.landmark[idx].y * h
                )

                left_eye.append((x, y))

            # RIGHT EYE
            for idx in RIGHT_EYE:

                x = int(
                    face_landmarks.landmark[idx].x * w
                )

                y = int(
                    face_landmarks.landmark[idx].y * h
                )

                right_eye.append((x, y))

            # EAR CALCULATION

            left_EAR = calculate_EAR(left_eye)

            right_EAR = calculate_EAR(right_eye)

            ear = (left_EAR + right_EAR) / 2

            # DROWSINESS LOGIC

            if ear < EAR_THRESHOLD:
                closed_frames += 1
            else:
                closed_frames = 0

            if closed_frames > FRAME_LIMIT:

                drowsy_detected = True

                save_log("Drowsiness Detected")

                cv2.putText(
                    frame,
                    "DROWSY!",
                    (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            # SHOW EAR

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

    # =====================================
    # SMOKING DETECTION (YOLOv26n)
    # =====================================

    smoking_detected = False

    smoke_results = smoking_model(
        frame,
        conf=0.40
    )

    for result in smoke_results:

        for box in result.boxes:

            cls = int(box.cls[0])

            class_name = result.names[cls]

            if class_name == "smoking":

                smoking_detected = True

                save_log("Smoking Detected")

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                conf = float(box.conf[0])

                # BLUE BOX

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"SMOKING {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )

    # =====================================
    # ACTIVE STATUS
    # =====================================

    if not drowsy_detected and not smoking_detected:

        cv2.putText(
            frame,
            "ACTIVE",
            (50, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # =====================================
    # FPS DISPLAY
    # =====================================

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (50, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
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