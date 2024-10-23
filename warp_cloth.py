import cv2
import numpy as np
import json
from PIL import Image



# Function to calculate scale and rotation based on body landmarks
def calculate_transformation(landmarks):
    Left_shoulder = landmarks['left_shoulder']  # Left shoulder landmark
    Right_shoulder = landmarks["right_shoulder"]  # Right shoulder landmark
    Left_hip = landmarks["left_hip"]  # Left hip landmark
    Right_hip = landmarks['right_hip']  # Right hip landmark

    # Distance between shoulders (for scaling)
    shoulder_width = np.linalg.norm(np.array(Left_shoulder) - np.array(Right_shoulder))
    
    # Mid-point of shoulders (for positioning)
    mid_shoulder = [(Left_shoulder[0] + Right_shoulder[0]) // 2,
                    (Left_shoulder[1] + Right_shoulder[1]) // 2]

    # Angle for rotation
    angle = np.arctan2(Right_shoulder[1] - Left_shoulder[1], Right_shoulder[0] - Left_shoulder[0])
    angle = np.degrees(angle)

    return shoulder_width, mid_shoulder, angle

with open('./user_landmarks.json', 'r') as fp:
    landmarks = json.load(fp)

    # Load clothing image
    clothing_img = cv2.imread('cloth_img1new.jpg')
    user_img = cv2.imread('usernew.jpg')

    # Calculate the transformation parameters (scale, position, rotation)
    # scale, position, rotation_angle = calculate_transformation(landmarks)
    scale, position, rotation_angle = 0.4, [0.0, 0.0], np.degrees(0)
    # print(scale, position, rotation_angle)

    # Apply transformations to the clothing image (scaling, rotating, translating)
    # Resize the clothing image
    clothing_img = cv2.resize(clothing_img, (int(clothing_img.shape[1] * scale), int(clothing_img.shape[0] * scale)))
    
    # clothing_img.resize((768,1024))
    cv2.imshow("a",clothing_img)
    
    # Rotate the clothing image
    matrix = cv2.getRotationMatrix2D((clothing_img.shape[1] // 2, clothing_img.shape[0] // 2), rotation_angle, 1)
    clothing_img = cv2.warpAffine(clothing_img, matrix, (clothing_img.shape[1], clothing_img.shape[0]))
    cv2.imwrite("warped_cloth.jpg", clothing_img)
    # cv2.imshow('Warped Image', Image.fromarray(clothing_img).show())

    # Overlay the clothing on the user image at the correct position
    # This part involves adding the clothing onto the user image with proper masking
    # print(Image.fromarray(clothing_img).split())
    # final_img = Image.fromarray(user_img).convert("RGBA")
    # final_img.paste(clothing_img, mask=Image.fromarray(clothing_img).split()[-1])
    