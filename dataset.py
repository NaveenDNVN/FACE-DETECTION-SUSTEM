import cv2
import numpy as np
import os
import pickle  # Used for saving our dataset

# Access the web camera
video = cv2.VideoCapture(0)  # 0 is for webcam
if not video.isOpened():  # Check if the camera opened successfully
    print("Error: Could not open webcam.")
    exit()

# Load the face detection classifier
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

face_data = []
name = input("Enter your name: ")

# Create the data directory if it does not exist
if not os.path.exists('data'):
    os.makedirs('data')

# To detect the face
while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Unable to capture video.")
        break  # Exit the loop if frame is not captured

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # for the frame of the detection
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resize_img = cv2.resize(crop_img, dsize=(50, 50))

        # Append the resized image to face_data
        face_data.append(resize_img)

        # Draw rectangle around detected face and display number of faces captured
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
        cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)

    cv2.imshow("Frame", frame)  # Show the frame with detected faces

    # Break the loop if 'q' is pressed or if 100 face samples are collected
    k = cv2.waitKey(1)
    if k == ord('q') or len(face_data) >= 100:
        break

# Release resources
video.release()
cv2.destroyAllWindows()

# Convert face_data into a numpy array and reshape it
face_data = np.array(face_data)
face_data = face_data.reshape(face_data.shape[0], -1)  # Reshape according to the number of samples collected

# Create dataset for student attendance and face detection
# Storing the student names
if 'names.pkl' not in os.listdir('data/'):
    names = [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    # Load existing names
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)

    # Ensure we only add names if we have less than 100
    while len(names) < 100:
        names.append(name)

    # Update the names.pkl file
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

# Storing face data for face detection
if 'face_data.pkl' not in os.listdir('data/'):  # Correct file name
    with open('data/face_data.pkl', 'wb') as f:  # Correct file name
        pickle.dump(face_data, f)
else:
    # Load existing face data
    with open('data/face_data.pkl', 'rb') as f:
        faces = pickle.load(f)

    faces = np.append(faces, face_data, axis=0)  # Append new faces to existing data

    # Update face_data.pkl file with new faces
    with open('data/face_data.pkl', 'wb') as f:
        pickle.dump(faces, f)

print("Face data collection complete.")














