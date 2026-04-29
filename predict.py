import cv2
import mediapipe as mp
import numpy as np
import pygame

# -------- MEDIAPIPE SETUP --------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# -------- SOUND --------
pygame.mixer.init()
pygame.mixer.music.load("drowsiness_model/alarm.mp3")
alarm_playing = False

# -------- EYE LANDMARKS --------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# -------- EAR FUNCTION --------
def calculate_EAR(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

# -------- VARIABLES --------
EAR_THRESHOLD = 0.25
FRAME_LIMIT = 15
closed_frames = 0

# -------- CAMERA --------
cap = cv2.VideoCapture(0)

print("MediaPipe Drowsiness Detection Started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    drowsy = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            h, w, _ = frame.shape

            left_eye = []
            right_eye = []

            # LEFT EYE
            for idx in LEFT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                left_eye.append((x, y))

            # RIGHT EYE
            for idx in RIGHT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                right_eye.append((x, y))

            # EAR calculation
            left_EAR = calculate_EAR(left_eye)
            right_EAR = calculate_EAR(right_eye)
            ear = (left_EAR + right_EAR) / 2

            # Detection logic
            if ear < EAR_THRESHOLD:
                closed_frames += 1
            else:
                closed_frames = 0

            if closed_frames > FRAME_LIMIT:
                drowsy = True

            # Show EAR
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # -------- STATUS --------
    if drowsy:
        cv2.putText(frame, "DROWSY!", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        if not alarm_playing:
            pygame.mixer.music.play(-1)
            alarm_playing = True
    else:
        cv2.putText(frame, "ACTIVE", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        if alarm_playing:
            pygame.mixer.music.stop()
            alarm_playing = False

    cv2.imshow("Drowsiness Detection (MediaPipe)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()