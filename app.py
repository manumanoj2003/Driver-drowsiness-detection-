from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from keras.models import load_model
import os
import signal

app = Flask(__name__)

# Load the trained model
model_path = "models/drowsiness_model.h5"
model = load_model(model_path)
labels = ["Open", "Closed"]

# Load face cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Constants
confidence_threshold = 0.4
drowsiness_score = 0
DROWSINESS_SCORE_THRESHOLD = 10
DROWSINESS_INCREMENT = 1
DROWSINESS_DECAY = 1


def preprocess_frame(frame):
    """Preprocesses the frame for prediction."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    reshaped = np.expand_dims(normalized, axis=-1)
    reshaped = np.expand_dims(reshaped, axis=0)
    return reshaped


def generate_frames():
    """Generates frames from the webcam and performs drowsiness detection."""
    global drowsiness_score
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not access the webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = frame[y:y + h, x:x + w]
            eye_roi = roi[int(h / 3):int(2 * h / 3), int(w / 4):int(3 * w / 4)]

            try:
                processed_roi = preprocess_frame(eye_roi)
                prediction = model.predict(processed_roi, verbose=0)
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

            except Exception as e:
                print(f"Error during prediction: {e}")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    """Renders the main HTML interface."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Provides the video feed to the browser."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/terminate', methods=['POST'])
def terminate():
    """Terminates the Flask application."""
    os.kill(os.getpid(), signal.SIGINT)
    return "Terminated", 200


if __name__ == "__main__":
    app.run(debug=True)
