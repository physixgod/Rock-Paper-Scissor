from flask import Flask, render_template, Response, jsonify
import cv2
import dlib
from flask_cors import CORS
from scipy.spatial import distance
import time

app = Flask(_name_)
CORS(app)

# Constants for drowsiness detection
DROWSINESS_THRESHOLD = 0.26
ANALYSIS_INTERVAL = 20  # Time interval for analyzing drowsiness (in seconds)
EAR_HISTORY = []        # List to store EAR values for analysis
LAST_ANALYSIS_TIME = time.time()
is_analysis_done = False  # Flag to track if analysis is done
total_average_ear = None  # Variable to store the total average EAR

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def detect_drowsiness():
    global EAR_HISTORY, LAST_ANALYSIS_TIME, is_analysis_done, total_average_ear

    cap = cv2.VideoCapture(0)
    while not is_analysis_done:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = hog_face_detector(gray)
        for face in faces:
            face_landmarks = dlib_facelandmark(gray, face)
            leftEye = []
            rightEye = []

            for n in range(36, 42):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                leftEye.append((x, y))
                next_point = n + 1
                if n == 41:
                    next_point = 36
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

            for n in range(42, 48):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                rightEye.append((x, y))
                next_point = n + 1
                if n == 47:
                    next_point = 42
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

            left_ear = calculate_EAR(leftEye)
            right_ear = calculate_EAR(rightEye)

            EAR = (left_ear + right_ear) / 2
            EAR = round(EAR, 2)
            EAR_HISTORY.append(EAR)
            if EAR < DROWSINESS_THRESHOLD:
                cv2.putText(frame, "DROWSY", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
                cv2.putText(frame, "Are you Sleepy?", (20, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                print("Drowsy")

        # Perform analysis after every ANALYSIS_INTERVAL seconds
        if time.time() - LAST_ANALYSIS_TIME >= ANALYSIS_INTERVAL:
            LAST_ANALYSIS_TIME = time.time()
            if EAR_HISTORY:  # Check if EAR_HISTORY is not empty
                total_average_ear = sum(EAR_HISTORY) / len(EAR_HISTORY)
                print("Total Average EAR:", total_average_ear)
                EAR_HISTORY = []  # Reset EAR history for the next analysis

                # Determine the drowsiness status based on the total average EAR
                result = "Drowsy" if total_average_ear < DROWSINESS_THRESHOLD else "Not Drowsy"

                # Set flag to indicate analysis is done
                is_analysis_done = True

                # Return the result as JSON response
                return jsonify({"result": result, "total_average_ear": total_average_ear})

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/drowsiness_status')
def drowsiness_status():
    global is_analysis_done, total_average_ear
    if is_analysis_done:  # Check if analysis is done
        result = "Drowsy" if total_average_ear < DROWSINESS_THRESHOLD else "Not Drowsy"
        is_analysis_done = False  # Reset the flag for the next analysis
        return jsonify({"result": result, "total_average_ear": total_average_ear})
    else:
        return jsonify({"result": "Analysis in progress"})

@app.route('/video_feed')
def video_feed():
    return Response(detect_drowsiness(), mimetype='multipart/x-mixed-replace; boundary=frame')


if _name_ == '_main_':
    app.run(debug=True)