from picamera2 import Picamera2
import cv2
import dlib
from scipy.spatial import distance
import time
import numpy as np
import os
from threading import Thread
from gpiozero import Servo
from time import sleep

# === Initialize servo on GPIO 14 ===
servo = Servo(14)
servo.detach()  # keep servo idle initially

# === Initialize camera ===
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# === Load detector and predictor ===
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# === Constants ===
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 15
YAWN_THRESH = 20
EMERGENCY_DELAY = 5  # seconds of continuous drowsiness

COUNTER = 0
last_yawn_time = 0
alarm_on = False
emergency_active = False
drowsy_start_time = None

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    return abs(np.mean(top_lip, axis=0)[1] - np.mean(low_lip, axis=0)[1])

def play_sound(file_path, loop=False):
    if loop:
        while alarm_on:
            os.system(f"aplay -q {file_path}")
    else:
        os.system(f"aplay -q {file_path}")

def start_drowsy_alarm():
    global alarm_on
    if not alarm_on:
        alarm_on = True
        Thread(target=play_sound, args=("drowsy.wav", True), daemon=True).start()

def stop_drowsy_alarm():
    global alarm_on
    alarm_on = False

def trigger_emergency():
    global emergency_active
    if not emergency_active:
        emergency_active = True
        print("🚨 EMERGENCY MODE ACTIVATED! Triggering servo...")
        servo.value = 1.0   # move to max position
        os.system("aplay -q emergency.wav")  # optional third sound
        sleep(1)
        servo.mid()         # move back to center
        sleep(0.5)
        servo.detach()
        print("Servo pulse complete.")
        emergency_active = False

while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 0)
    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

        leftEye = shape[36:42]
        rightEye = shape[42:48]
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        lip_dist = lip_distance(shape)

        # Draw indicators
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [shape[48:60]], -1, (0, 255, 0), 1)

        # === Drowsy Detection ===
        if ear < EYE_AR_THRESH:
            if drowsy_start_time is None:
                drowsy_start_time = time.time()
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                print("⚠️ Driver is drowsy!")
                start_drowsy_alarm()

                # Check for emergency condition
                if time.time() - drowsy_start_time > EMERGENCY_DELAY:
                    trigger_emergency()
        else:
            COUNTER = 0
            stop_drowsy_alarm()
            drowsy_start_time = None

        # === Yawn Detection ===
        if lip_dist > YAWN_THRESH and (time.time() - last_yawn_time > 5):
            print("😮 Driver is yawning!")
            Thread(target=play_sound, args=("yawn.wav",), daemon=True).start()
            last_yawn_time = time.time()

        # Display data
        cv2.putText(frame, f"EAR: {ear:.2f}", (400, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"YAWN: {lip_dist:.2f}", (400, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Driver Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stop_drowsy_alarm()
cv2.destroyAllWindows()
picam2.stop()
servo.detach()
