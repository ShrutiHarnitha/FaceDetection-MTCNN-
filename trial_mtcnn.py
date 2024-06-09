import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# FACE DETECTION

# Initialize the MTCNN detector
detector = MTCNN()
#detector = MTCNN(min_face_size=20, scale_factor=0.7, thresholds=[0.6, 0.7, 0.7])

# Load the input image
img = cv2.imread('media/faces_pic2.jpg')

# Detect faces in the image
faces = detector.detect_faces(img)

# Check if any faces are detected
if faces:
    # Draw rectangles around the detected faces
    for result in faces:
        x, y, w, h = result['box']
        x1, y1 = x + w, y + h
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)

    # Resize the output image to make it smaller
    scale_percent = 70  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Display the resized output image with detected faces
    cv2.imshow('Detected Faces (Resized)', resized_img)
else:
    # Resize the original image if no faces are detected
    scale_percent = 50  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Display the resized original image if no faces are detected
    cv2.imshow('No Faces Detected (Resized)', resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()



# FACE EXTRACTION

# Iterate over all detected faces
# for i, result in enumerate(faces):
#     x, y, w, h = result['box']
#     # Extract the face region
#     my_face = img[y:y+h, x:x+w]
    
#     # Resize the face image to 160x160 pixels as required by FaceNet
#     my_face = cv2.resize(my_face, (160, 160))
    
#     # Display the face using matplotlib
#     plt.subplot(1, len(faces), i+1)  # Create a subplot for each face
#     plt.imshow(cv2.cvtColor(my_face, cv2.COLOR_BGR2RGB))
#     plt.axis('off')

# plt.show()

