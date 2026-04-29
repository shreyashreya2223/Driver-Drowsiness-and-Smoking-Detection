from ultralytics import YOLO
import cv2

model = YOLO("smoke_detection/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated = results[0].plot()

    if len(results[0].boxes) > 0:
        cv2.putText(annotated, "SMOKING DETECTED!", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.imshow("Smoking Detection", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()