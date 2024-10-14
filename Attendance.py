import cv2
import numpy as np
import os
import csv  # To handle attendance of the student
import time  # For storing the timing of the student
import pickle
from sklearn.neighbors import KNeighborsClassifier  # For training the dataset using KNN
from datetime import datetime  # To get date and time for the student attendance

# Accessing the web camera
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not access webcam.")
    exit()

facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Importing faces and labels from pickle
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)  # For the names of the students

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)  # For the faces of the students

# Using KNN algorithm for face recognition
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)  # Train KNN with faces and labels

imgbackground = cv2.imread("bgi.jpg")

# Creating column names for the attendance CSV
COL_NAMES = ['NAME', 'TIME']

# Start face detection and recognition
while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Unable to capture video.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resize_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resize_img)  # Predict the face using KNN

        # Timestamp for the attendance
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        # Check if the CSV file for the day exists
        attendance_file = f"Attendance/Attendance_{date}.csv"
        file_exists = os.path.isfile(attendance_file)

        # Draw rectangle around the face and label the output name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Create attendance record
        attendance = [str(output[0]), str(timestamp)]

        # Insert the frame into the background
        imgbackground[162:162 + 480, 55:55 + 640] = frame

        # Show the frame with background and recognized face
        cv2.imshow("Frame", imgbackground)

        # Check for the key press to take attendance or quit
        k = cv2.waitKey(1)

        if k == ord('o'):  # If 'o' is pressed, take attendance
            time.sleep(1)  # Short delay before taking attendance

            if file_exists:
                # Append to existing attendance file
                with open(attendance_file, "a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(attendance)
            else:
                # Create new attendance file and add column headers
                with open(attendance_file, "w") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(COL_NAMES)
                    writer.writerow(attendance)

        # Exit the program if 'q' is pressed
        if k == ord('q'):
            break

# Release video capture and close all windows
video.release()
cv2.destroyAllWindows()







