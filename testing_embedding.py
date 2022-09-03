# this is a simple test file used to check if the pose embedding is porperly working

import mediapipe as mp
import numpy as np
import cv2

from pose_embedder import CustomPoseEmbedder

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles


img = cv2.imread('data/train/pose_1/test_image.jpg')

cv2.imshow('win',img)


with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
    # Convert the BGR image to RGB and process it with MediaPipe Pose.
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Print nose landmark.
    image_hight, image_width, _ = img.shape
    if not results.pose_landmarks:
        pass
    else:
        
        pose_landmarks = np.array([[lmk.x * img.shape[1], lmk.y * img.shape[0], lmk.z * img.shape[1]]
                                 for lmk in results.pose_landmarks.landmark], dtype=np.float32)
        print(pose_landmarks)
        
        my_pose = CustomPoseEmbedder()
        data = my_pose(pose_landmarks)
        print(data)

        # Draw pose landmarks.
        annotated_image = img.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imshow('my_win',annotated_image)
        # cv2.imwrite('img.jpg',annotated_image)
        cv2.waitKey(0)