from picamera2 import Picamera2
import cv2
import dlib
from scipy.spatial import distance
import time
import numpy as np
import os
from threading import Thread
from gpiozero import OutputDevice
from time import sleep

# === Initialize relay on GPIO 14 ===
# Note: If your relay is active-low, change active_high to False
relay = OutputDevice(14, active_high=False, initial_value=False)
print(f"Relay initialized. State: {relay.value}")
print("Relay should be OFF now. If it's ON, there may be a hardware issue.")

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
EMERGENCY_DELAY = 2  # seconds of continuous drowsiness (changed from 5 to 2)

COUNTER = 0
last_yawn_time = 0
alarm_on = False
emergency_active = False
emergency_start_time = None
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
    global emergency_active, emergency_start_time
    if not emergency_active:
        emergency_active = True
        emergency_start_time = time.time()
        print("🚨 EMERGENCY MODE ACTIVATED! Triggering relay...")
        print(f"Relay state before ON: {relay.value}")
        relay.on()  # activate relay
        print(f"Relay state after ON: {relay.value}")
        os.system("aplay -q emergency.wav")  # play emergency sound
        print("Emergency relay activated.")

while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # === Emergency Mode Auto-Deactivate (check every frame) ===
    if emergency_active and time.time() - emergency_start_time > 3:
        print(f"Relay state before OFF: {relay.value}")
        relay.off()
        print(f"Relay state after OFF: {relay.value}")
        emergency_active = False
        emergency_start_time = None
        print("Emergency deactivated after 3 seconds.")

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
                cv2.putText(frame, "⚠️ DROWSY!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                print("⚠️ Driver is drowsy!")
                start_drowsy_alarm()

                # Check for emergency condition
                if time.time() - drowsy_start_time > EMERGENCY_DELAY and not emergency_active:
                    trigger_emergency()
        else:
            COUNTER = 0
            stop_drowsy_alarm()
            drowsy_start_time = None

        # === Emergency Mode Display ===
        if emergency_active:
            cv2.putText(frame, "🚨 EMERGENCY!", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # === Yawn Detection ===
        if lip_dist > YAWN_THRESH and (time.time() - last_yawn_time > 5):
            cv2.putText(frame, "😮 YAWNING!", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
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
relay.off()  # ensure relay is off when program ends
relay.close()