from distutils.errors import PreprocessError
import cv2
import numpy as np 
import keras
from keras.models import load_model
import sys
import csv
import os
sys.path.append('/home/pinchuan/api/openpose/ultralytics')
from ultralytics import YOLO
from collections import defaultdict

sys.path.append('/home/pinchuan/api/openpose/build/python/')
from openpose import pyopenpose as op


# YOLOv8 model
model = YOLO('yolov8s.pt') 
class_names = ['person']




n_steps = 16 # 32 timesteps per series

iterator=0
openpose_output=[]   #Will store the openpose time series data for recent n_steps
inteval = 1
sequence_start=0   #starting location of circular array # starting location of circular array

# OpenPose params
params = dict()
params["model_folder"] = "/home/pinchuan/api/openpose/models3"

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Specify the folder containing the videos
videos_folder = "/home/pinchuan/Videos/0322exp05/archive/0325"

# Open CSV file for writing
csv_filename = '/home/pinchuan/Videos/0322exp05/archive/basket1.csv'
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Video_Name', 'Person_ID', 'Keypoints'])

# Iterate over all video files in the folder
for video_file in os.listdir(videos_folder):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(videos_folder, video_file)

        # Extract video name without extension
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Open video
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # video_writer = cv2.VideoWriter(f'output_{video_name}.avi', fourcc, 16, (640, 480))



        cap = cv2.VideoCapture(video_path)
        frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # Store the track history
        track_history = defaultdict(lambda: []) 
        top, bottom, left, right = 10, 10, 20, 20  
        border_color = 0

        frame_counter = 0  # Add a frame counter

        while cap.isOpened() and frame_counter < 16:  # Stop after processing 16 frames
            success, frame = cap.read()

            if success:
                # OpenPose 
                datum = op.Datum()
                datum.cvInputData = frame
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))  
                frame = datum.cvOutputData
                results = model.track(frame, classes=0, persist=True)
                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()  
                # 提取OpenPose的關節點資訊
                keypoints_list = datum.poseKeypoints


                points = []
                if keypoints_list is not None and len(keypoints_list) > 0:
                    for person_id, keypoints in enumerate(keypoints_list):
                        
                        if keypoints.any():
                        
                        # 移除confidence score
                            keypoints = keypoints[:,:-1]  
                        print("Person {}: {}".format(person_id, keypoints))


                        row = [video_name, person_id]

                
                                # 構建資料行
                            
                        for i, keypoint in enumerate(keypoints):
                                    x = keypoint[0]
                                    y = keypoint[1]

                                    row.insert(2*i+2, x)  
                                    row.insert(2*i+3, y)

                                # 寫入資料行  
                        # Write data row to CSV
                        csv_writer.writerow(row)

                frame_counter += 1  # Increment frame counter
            else:
                        break  # Break the loop when the video ends
    if keypoints_list is None or not any(keypoints.any() for keypoints in keypoints_list):
    # 如果在影片中沒有檢測到任何物體,就跳過此影片
        print(f"No objects detected in this video, skipping to next video.")
        continue

    keypoints_sequence = []
    for person_id, keypoints in enumerate(keypoints_list):
            if keypoints.any():
                # Assuming keypoints are being tracked at an interval of n_steps
                if frame_num % n_steps == 0:
                    track_history[person_id].append(keypoints)  # Storing keypoints for prediction
                    if len(track_history[person_id]) > n_steps:
                        track_history[person_id] = track_history[person_id][-n_steps:]
                        
                        flattened_keypoints = PreprocessError(keypoints)
                        keypoints_sequence.append(flattened_keypoints)




            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            # annotated_frame = cv2.resize(annotated_frame, (200, 200))

            # out = cv2.VideoWriter('output888.avi', fourcc, 20, (640, 480))
            # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.imshow('Pose Tracking',annotated_frame)
        
        
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# out.write(frame)          
cap.release()
cv2.destroyAllWindows()





