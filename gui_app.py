import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import pygame
import time
import os
import matplotlib.pyplot as plt

# -------- UI SETTINGS --------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# -------- MODELS --------
smoke_model = YOLO("smoke_detection/best.pt")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

def calculate_EAR(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A+B)/(2.0*C)

# -------- SOUND --------
pygame.mixer.init()
pygame.mixer.music.load("drowsiness_model/alarm.mp3")
alarm_playing = False

# -------- VARIABLES --------
cap = None
running = False
closed_frames = 0
frame_count = 0
smoking_detected = False

EAR_THRESHOLD = 0.25
FRAME_LIMIT = 10

# -------- GRAPH DATA --------
score_history = []
time_history = []
start_time = time.time()

# -------- TRACKING --------
drowsy_count = 0
smoking_count = 0
drowsy_flag = False
smoking_flag = False
score = 100

prev_time = time.time()

os.makedirs("alerts", exist_ok=True)

# -------- GRAPH FUNCTION --------
def show_graph():
    if len(score_history) == 0:
        print("No data yet!")
        return

    plt.figure(figsize=(6,4))
    plt.plot(time_history, score_history, color='blue')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Driver Score")
    plt.title("Driver Behavior Over Time")
    plt.grid(True)
    plt.show()

# -------- CAMERA LOOP --------
def update_frame():
    global cap, running, closed_frames, frame_count
    global smoking_detected, alarm_playing
    global drowsy_count, smoking_count
    global drowsy_flag, smoking_flag
    global score, prev_time

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.resize(frame, (700, 500))
    frame = cv2.flip(frame, 1)

    # -------- FPS --------
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time))
    prev_time = curr_time

    # -------- SMOKING --------
    frame_count += 1
    if frame_count % 5 == 0:
        results = smoke_model(frame, conf=0.4)
        smoking_detected = len(results[0].boxes) > 0
        annotated = results[0].plot()
    else:
        annotated = frame.copy()

    # -------- DROWSINESS --------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    drowsy = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            left_eye = [(int(face_landmarks.landmark[i].x*w),
                         int(face_landmarks.landmark[i].y*h)) for i in LEFT_EYE]

            right_eye = [(int(face_landmarks.landmark[i].x*w),
                          int(face_landmarks.landmark[i].y*h)) for i in RIGHT_EYE]

            ear = (calculate_EAR(left_eye)+calculate_EAR(right_eye))/2

            if ear < EAR_THRESHOLD:
                closed_frames += 1
            else:
                closed_frames = 0

            if closed_frames > FRAME_LIMIT:
                drowsy = True

            cv2.putText(annotated, f"EAR: {ear:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # -------- STATUS --------
    if smoking_detected and drowsy:
        status = "SMOKING + DROWSY"
        color = "red"
    elif smoking_detected:
        status = "SMOKING"
        color = "orange"
    elif drowsy:
        status = "DROWSY"
        color = "red"
    else:
        status = "SAFE"
        color = "green"

    status_label.configure(text=status, text_color=color)

    # -------- ALERT LOGIC --------
    if drowsy:
        if not drowsy_flag:
            drowsy_count += 1
            score -= 10
            drowsy_flag = True
    else:
        drowsy_flag = False

    if smoking_detected:
        if not smoking_flag:
            smoking_count += 1
            score -= 5
            smoking_flag = True
    else:
        smoking_flag = False

    # -------- RECOVERY --------
    if not drowsy and not smoking_detected:
        score += 0.2

    score = max(0, min(100, score))

    # -------- STORE GRAPH DATA --------
    current_time = time.time() - start_time
    score_history.append(score)
    time_history.append(current_time)

    if len(score_history) > 200:
        score_history.pop(0)
        time_history.pop(0)

    # -------- SAVE + ALERT --------
    if smoking_detected or drowsy:
        filename = f"alerts/{int(time.time()*1000)}.jpg"
        cv2.imwrite(filename, annotated)

        with open("log.txt", "a") as f:
            f.write(f"{status} at {time.ctime()}\n")

        if not alarm_playing:
            pygame.mixer.music.play(-1)
            alarm_playing = True
    else:
        if alarm_playing:
            pygame.mixer.music.stop()
            alarm_playing = False

    # -------- UPDATE UI --------
    fps_label.configure(text=f"FPS: {fps}")
    accuracy_label.configure(text="Accuracy: 85.7%")
    score_label.configure(text=f"Driver Score: {int(score)}")

    # -------- DISPLAY --------
    img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    app.after(15, update_frame)

# -------- BUTTONS --------
def start():
    global running, cap
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        update_frame()

def stop():
    global running
    running = False
    if cap:
        cap.release()

# -------- GUI --------
app = ctk.CTk()
app.title("Driver Monitoring System")
app.geometry("900x750")

title = ctk.CTkLabel(app, text="Driver Monitoring System",
                     font=("Arial", 26, "bold"))
title.pack(pady=10)

video_label = ctk.CTkLabel(app, text="")
video_label.pack(pady=10)

status_label = ctk.CTkLabel(app, text="SAFE",
                           font=("Arial", 22, "bold"))
status_label.pack(pady=5)

fps_label = ctk.CTkLabel(app, text="FPS: 0")
fps_label.pack()

accuracy_label = ctk.CTkLabel(app, text="Accuracy: 85.7%")
accuracy_label.pack()

score_label = ctk.CTkLabel(app, text="Driver Score: 100",
                          font=("Arial", 18, "bold"))
score_label.pack(pady=5)

btn_frame = ctk.CTkFrame(app)
btn_frame.pack(pady=10)

start_btn = ctk.CTkButton(btn_frame, text="Start", command=start, width=140)
start_btn.grid(row=0, column=0, padx=10)

stop_btn = ctk.CTkButton(btn_frame, text="Stop", command=stop, width=140)
stop_btn.grid(row=0, column=1, padx=10)

graph_btn = ctk.CTkButton(btn_frame, text="Show Graph", command=show_graph, width=140)
graph_btn.grid(row=0, column=2, padx=10)

app.mainloop()