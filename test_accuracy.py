import os
import cv2
from ultralytics import YOLO

model = YOLO("smoke_detection/best.pt")

smoking_path = "test_images/smoking"
nonsmoking_path = "test_images/not_smoking"

TP = TN = FP = FN = 0

# -------- SMOKING IMAGES --------
for img_name in os.listdir(smoking_path):
    img_path = os.path.join(smoking_path, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    results = model(img)
    detected = len(results[0].boxes) > 0

    if detected:
        TP += 1   # correct smoking
    else:
        FN += 1   # missed smoking

# -------- NON-SMOKING IMAGES --------
for img_name in os.listdir(nonsmoking_path):
    img_path = os.path.join(nonsmoking_path, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    results = model(img)
    detected = len(results[0].boxes) > 0

    if detected:
        FP += 1   # false alarm
    else:
        TN += 1   # correct non-smoking

# -------- METRICS --------
total = TP + TN + FP + FN
accuracy = (TP + TN) / total

print("\n--- RESULTS ---")
print("TP (Smoking correct):", TP)
print("TN (Non-smoking correct):", TN)
print("FP (Wrong smoking):", FP)
print("FN (Missed smoking):", FN)

print("\nAccuracy:", round(accuracy * 100, 2), "%")