import cv2
import dlib
import numpy as np
import os
import csv

image_paths = 'Final Project/Video Model/Frames/'
# Load the face detector and the landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define the facial landmarks we are interested in
left_eye_pts = [36, 37, 38, 39, 40, 41]
right_eye_pts = [42, 43, 44, 45, 46, 47]
nose_pts = [29, 30, 31, 32, 33, 34, 35]
mouth_pts = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]

# Define the constants for the dimensional approach
arousal_const = 0.45
valence_const = 0.35
neutral_const = 1 - arousal_const - valence_const

# Define the function to calculate the arousal and valence
def calculate_arousal_valence(brow_dist, nose_width, mouth_height, mouth_nose_dist):
    # Calculate the normalized feature vectors
    brow_dist_norm = brow_dist / np.linalg.norm(brow_dist)
    nose_width_norm = nose_width / np.linalg.norm(nose_width)
    mouth_height_norm = mouth_height / np.linalg.norm(mouth_height)
    mouth_nose_dist_norm = mouth_nose_dist / np.linalg.norm(mouth_nose_dist)

    # Calculate the arousal and valence using the dimensional approach
    arousal = arousal_const * mouth_height_norm + (1 - arousal_const) * mouth_nose_dist_norm
    valence = valence_const * nose_width_norm - (1 - valence_const) * brow_dist_norm

    # Normalize the arousal and valence values
    arousal = (arousal + 1) / 2
    valence = (valence + 1) / 2

    return arousal, valence

# Define the function to extract facial features from an image
def extract_features(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    # Loop through each face detected
    for face in faces:
        # Get the landmarks for the face
        landmarks = predictor(gray, face)

        # Extract the left and right eye, nose, and mouth landmarks
        left_eye = np.array([(landmarks.part(pt).x, landmarks.part(pt).y) for pt in left_eye_pts])
        right_eye = np.array([(landmarks.part(pt).x, landmarks.part(pt).y) for pt in right_eye_pts])
        nose = np.array([(landmarks.part(pt).x, landmarks.part(pt).y) for pt in nose_pts])
        mouth = np.array([(landmarks.part(pt).x, landmarks.part(pt).y) for pt in mouth_pts])

        # Calculate the distance between the eyebrows and the nose
        brow_dist = nose[0][1] - (left_eye[1][1] + right_eye[1][1]) / 2

        # Calculate the width of the nose
        nose_width = nose[-1][0] - nose[0][0]

        # Calculate the height of the mouth
        mouth_height = mouth[-1][1] - mouth[0][1]

        # Calculate the distance between the mouth and the nose
        mouth_nose_dist = nose[3][1] - mouth[3][1]

        # Calculate the arousal and valence
        arousal, valence = calculate_arousal_valence(brow_dist, nose_width, mouth_height, mouth_nose_dist)

        # Return the arousal and valence values
        return arousal, valence

# Define the output CSV file path and fieldnames
output_csv = 'Final Project/Video Model/Train.csv'
fieldnames = ['image_path', 'brow_dist', 'nose_width', 'mouth_height', 'mouth_nose_dist', 'arousal', 'valence']

# Open the output CSV file and write the header row
with open(output_csv, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Loop through each input image
    for image_path in image_paths:
        # Extract the facial features and arousal/valence values
        brow_dist, nose_width, mouth_height, mouth_nose_dist, arousal, valence = extract_features(image_path)

        # Write the row to the output CSV file
        writer.writerow({'image_path': image_path, 'brow_dist': brow_dist, 'nose_width': nose_width, 'mouth_height': mouth_height, 'mouth_nose_dist': mouth_nose_dist, 'arousal': arousal, 'valence': valence})