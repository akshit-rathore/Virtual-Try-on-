import cv2
import sys
import os
import numpy as np
import cv2
import mediapipe as mp
import json
from PIL import Image

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


image_path = r'C:\React\VTON\usernew.jpg' 
user_image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(user_image, cv2.COLOR_BGR2RGB)

# Pose Detection
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        
    results = pose.process(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )

    cv2.imwrite("user_pose.jpg", image_bgr)
    # cv2.imshow('Pose Detection on Image', image_bgr)


# Extracting user landmarks
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
results = pose.process(image_rgb)
if results.pose_landmarks:
    print("User Pose landmarks detected!")
else:
    print("No pose landmarks found.")


# Define a function to extract key landmarks
def extract_key_landmarks(landmarks):
    key_landmarks = {}
    
    # List of landmark indices for key body parts
    landmark_names = {
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_hip': 23,
        'right_hip': 24,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16
    }
    
    # Extract the key landmarks and store in a dictionary
    for name, index in landmark_names.items():
        landmark = landmarks.landmark[index]
        key_landmarks[name] = (landmark.x, landmark.y)  # (x, y, z) coordinates
    
    return key_landmarks

# Extract user key landmarks from the detected pose landmarks
if results.pose_landmarks:
    user_key_landmarks = extract_key_landmarks(results.pose_landmarks)
    
    # Print out the extracted key landmarks
    for name, coords in user_key_landmarks.items():
        print(f"{name}: {coords}")
else:
    print("No landmarks detected.")

# # Visualize key landmarks by drawing circles on the image
def draw_user_key_landmarks(image, key_landmarks):
    height, width, _ = image.shape
    for name, coords in key_landmarks.items():
        x, y = int(coords[0] * width), int(coords[1] * height)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle
    

# Visualize the user key landmarks on the original user image
draw_user_key_landmarks(user_image, user_key_landmarks)
cv2.imwrite("user_key_landmarks.jpg", user_image)

def denormalize_landmarks(landmarks, image_width, image_height):
    # Scale the normalized landmarks to pixel coordinates
    denormalized_landmarks = {}
    for lm in landmarks.keys():
        x_pixel = int(landmarks[lm][0] * image_width)  # Convert x from [0, 1] to pixel
        y_pixel = int(landmarks[lm][1] * image_height)  # Convert y from [0, 1] to pixel
        denormalized_landmarks[lm] = (x_pixel, y_pixel)
    return denormalized_landmarks

denormalized_landmarks = denormalize_landmarks(user_key_landmarks,768,1024)
# print(denormalized_landmarks)


image_path = r'C:\React\VTON\cloth_img1new.jpg'  # Specify your image path
cloth_image = cv2.imread(image_path)

gray_cloth_img = cv2.cvtColor(cloth_image, cv2.COLOR_BGR2GRAY)
# sift = cv2.SIFT_create()
# keypoints, descriptors = sift.detectAndCompute(gray_cloth_img, None)

_, thresh = cv2.threshold(gray_cloth_img, 230, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) == 0:
    raise ValueError("No cloth found in the image.")

# Assume the largest contour is the cloth
cloth_contour = max(contours, key=cv2.contourArea)

# Step 4: Find the bounding rectangle to approximate key areas of the cloth
x, y, w, h = cv2.boundingRect(cloth_contour)

# Step 5: Define key landmark points manually
# Collar left and right
collar_left = (x, y)
collar_right = (x + w, y)
    
# Sleeve left and right
sleeve_left = (x, y + int(h * 0.3))
sleeve_right = (x + w, y + int(h * 0.3))
    
# Bottom left and right
bottom_left = (x, y + h)
bottom_right = (x + w, y + h)

# Collect the key landmarks
cloth_key_landmarks = {"collar_left":tuple(collar_left), 
                   "collar_right":tuple(collar_right), 
                   "sleeve_left":tuple(sleeve_left), 
                   "sleeve_right":tuple(sleeve_right), 
                   "bottom_left":tuple(bottom_left), 
                   "bottom_right":tuple(bottom_right)}
print("Cloth Pose landmarks detected!\n",cloth_key_landmarks)

def draw_cloth_key_landmarks(image, key_landmarks):
    height, width, _ = image.shape
    for name, coords in key_landmarks.items():
        x, y = int(coords[0]), int(coords[1])
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle

draw_cloth_key_landmarks(cloth_image, cloth_key_landmarks)
cv2.imwrite("cloth_key_landmarks.jpg", cloth_image)








# Visualize the user key landmarks on the original cloth image
# cloth_image_with_kp = cv2.drawKeypoints(cloth_image, keypoints, None, color=(0, 255, 0))
# cv2.imwrite("cloth_key_landmarks.jpg", cloth_image_with_kp)

with open('./user_landmarks.json', 'w') as fp:
    json.dump(user_key_landmarks, fp)
with open('./cloth_landmarks.json', 'w') as fp:
    json.dump(cloth_key_landmarks, fp)


# Display the user and cloth image with key landmarks drawn
# cv2.imshow('User Image with Key Landmarks', user_image)
# cv2.imshow('Cloth Image with Key Landmarks', cloth_image_with_kp)

# Release resources
cv2.waitKey(0)
cv2.destroyAllWindows()
       
