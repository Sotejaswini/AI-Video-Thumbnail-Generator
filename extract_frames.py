import cv2
import os

video_dir = "videos"
output_dir = "original_frames"
os.makedirs(output_dir, exist_ok=True)

for video in os.listdir(video_dir):
    if video.endswith(".mp4"):
        cap = cv2.VideoCapture(os.path.join(video_dir, video))
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_dir, f"{video}_original.jpg")
            cv2.imwrite(output_path, frame)
        cap.release()
