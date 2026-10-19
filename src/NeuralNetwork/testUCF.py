import cv2
import os

video_path = "../Downloads/action_youtube_naudio/walking/v_walk_dog_01/001.avi"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video")
else:
    print("Video opened successfully")
    print("Frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("FPS:", cap.get(cv2.CAP_PROP_FPS))
    print("Width:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("Height:", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

cap.release()