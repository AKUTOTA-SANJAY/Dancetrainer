import os
os.environ["OMP_NUM_THREADS"] = "1"


from sklearn.cluster import KMeans

import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_key_frames(video_path, output_folder, num_key_frames=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    
    features = np.zeros((frame_count, frame_width * frame_height), dtype=np.float32)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features[frame_idx] = gray_frame.flatten()

        frame_idx += 1

    cap.release()

 
    kmeans = KMeans(n_clusters=num_key_frames, random_state=0)
    kmeans.fit(features)

   
    key_frame_indices = []
    for centroid in kmeans.cluster_centers_:
        
        distances = np.linalg.norm(features - centroid, axis=1)
        
        closest_idx = np.argmin(distances)
        key_frame_indices.append(closest_idx)

   
    cap = cv2.VideoCapture(video_path)
    for idx in key_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))  
        _, frame = cap.read()
        cv2.imwrite(f"{output_folder}/frame_{idx}.jpg", frame)
        print(f"Key frame {idx} extracted successfully.")

    cap.release()




video_path = "C:/Users/hp/Downloads/DanceApp/dance_videos/tutor.mp4"
output_folder = "keysframes"



extract_key_frames(video_path, output_folder)
