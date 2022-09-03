# This file is used to do all the required pre-preocessing steps
# It also generates the required .npy files uisng the input videos


import mediapipe as mp
import numpy as np
import cv2

from pose_embedder import CustomPoseEmbedder


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles


my_data = []                            # this will later be converted to a numpy array to be saved as a .npy binary file.
steps = 30                              # this the number of frames that is taken from each video
data_location = 'data/test/'           # this is the location of the folder for the data
pose_selected = "pose_3"                # this is the pose name

for i in range(1,2):
    cap = cv2.VideoCapture(data_location + pose_selected +'/'+ pose_selected +'_'+ str(i)+'.mp4')
    
    

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = num_frames/steps
    frames_added = 0
    
    # result = cv2.VideoWriter(pose_selected+'_3_embedded.mp4', cv2.VideoWriter_fourcc(*'MJPG'),7.0, (int(width),int(height)))
    
    temp_data = []
    
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
        # this is used to calculate the normalized and video size independant pose data
        my_pose = CustomPoseEmbedder()
        
        for j in range(num_frames):
            ret,img = cap.read()
            
            # stop the loop if the end of video is reached.
            # this is mostly not required as the for loop only runs in the range
            if ret == False:
                break
            
            else:
                if j == int(frame_step*frames_added):
                    frames_added = frames_added+1
            
                    # Convert the BGR image to RGB and process it with MediaPipe Pose.
                    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    
                    if not results.pose_landmarks:
                        pass
                    else:
                        
                        pose_landmarks = np.array([[lmk.x*width, lmk.y*height, lmk.z*width]
                                                for lmk in results.pose_landmarks.landmark], dtype=np.float32)
                        
                        data = my_pose(pose_landmarks)
                        data = data[:,:2]
                        data = data.reshape((50))
                        # print(data.shape)
                        data = data.tolist()
                        
                        temp_data.append(data)

                        # Draw pose landmarks.
                        annotated_image = img.copy()
                        mp_drawing.draw_landmarks(
                            annotated_image,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                        
                        # result.write(annotated_image)
                        
                        cv2.imshow('my_win',annotated_image)
                        cv2.waitKey(1)
            
                # print(frames_added)
            
        # result.release() 
            
    if len(temp_data) == 30:
        my_data.append(temp_data)
    else:
        print("video number {} was not added to the numpy array".format(i))
    
array_to_save = np.array(my_data,dtype=np.float32)
array_to_save = np.around(array_to_save,6)

print(array_to_save)
print(array_to_save.shape)

print(np.max(array_to_save[0]),np.max(array_to_save[1]))
print(np.min(array_to_save[0]),np.min(array_to_save[1]))

# saves the data to the same folder as the training video files
np.save(data_location + pose_selected +'/'+ pose_selected +'_test',array_to_save)