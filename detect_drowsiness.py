import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model_path = "models/drowsiness_model.h5"
model = load_model(model_path)

# Constants
DROWSINESS_SCORE_THRESHOLD = 10
DROWSINESS_INCREMENT = 1
DROWSINESS_DECAY = 1
labels = ["Open", "Closed"]
confidence_threshold = 0.4

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

drowsiness_score = 0

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    reshaped = np.expand_dims(normalized, axis=-1)
    reshaped = np.expand_dims(reshaped, axis=0)
    return reshaped

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        continue

    x, y, w, h = faces[0]
    roi = frame[y:y + h, x:x + w]
    eye_roi = roi[int(h / 3):int(2 * h / 3), int(w / 4):int(3 * w / 4)]

    # Visualize cropped eye region
    cv2.imshow("Eye Region", eye_roi)

    try:
        processed_roi = preprocess_frame(eye_roi)
        prediction = model.predict(processed_roi, verbose=0)
    except Exception as e:
        print(f"Error during prediction: {e}")
        continue

    class_index = np.argmax(prediction)
    if prediction[0][class_index] > confidence_threshold:
        label = labels[class_index]
    else:
        label = "Uncertain"

    if class_index == 1:  # Closed eyes
        drowsiness_score += DROWSINESS_INCREMENT
    else:
        drowsiness_score = max(0, drowsiness_score - DROWSINESS_DECAY)

    if drowsiness_score > DROWSINESS_SCORE_THRESHOLD:
        cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.putText(frame, f"State: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Score: {drowsiness_score}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(frame, (x + int(w / 4), y + int(h / 3)), 
                  (x + int(3 * w / 4), y + int(2 * h / 3)), (0, 0, 255), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
