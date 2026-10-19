import cv2
import os
import math
import csv
from ultralytics import YOLO

dataset_path = "action_youtube_naudio"
output_path = "ucf11_dataset.csv"

model = YOLO("yolo11n.pt")

with open(output_path, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "video",
        "action",
        "average_movement",
        "maximum_movement",
        "detected_frames",
        "total_frames"
    ])

    for action in os.listdir(dataset_path):
        action_path = os.path.join(dataset_path, action)

        if not os.path.isdir(action_path):
            continue

        if action == "Annotation":
            continue

        print("Processing action:", action)

        for root, dirs, files in os.walk(action_path):
            for filename in files:
                if not filename.lower().endswith(".avi"):
                    continue

                video_path = os.path.join(root, filename)
                cap = cv2.VideoCapture(video_path)

                if not cap.isOpened():
                    print("Could not open:", filename)
                    continue

                positions = []
                total_frames = 0
                detected_frames = 0

                while True:
                    ret, frame = cap.read()

                    if not ret:
                        break

                    results = model(frame, verbose=False)
                    frame_position = None

                    for result in results:
                        if result.boxes is None:
                            continue

                        for box in result.boxes:
                            if int(box.cls[0]) != 0:
                                continue

                            x1, y1, x2, y2 = box.xyxy[0]

                            center_x = (float(x1) + float(x2)) / 2
                            center_y = (float(y1) + float(y2)) / 2

                            frame_position = (center_x, center_y)
                            break

                        if frame_position is not None:
                            break

                    positions.append(frame_position)

                    if frame_position is not None:
                        detected_frames += 1

                    total_frames += 1

                cap.release()

                movements = []

                for i in range(1, len(positions)):
                    previous = positions[i - 1]
                    current = positions[i]

                    if previous is None or current is None:
                        continue

                    distance = math.sqrt(
                        (current[0] - previous[0]) ** 2
                        + (current[1] - previous[1]) ** 2
                    )

                    movements.append(distance)

                average_movement = 0
                maximum_movement = 0

                if len(movements) > 0:
                    average_movement = sum(movements) / len(movements)
                    maximum_movement = max(movements)

                writer.writerow([
                    filename,
                    action,
                    round(average_movement, 2),
                    round(maximum_movement, 2),
                    detected_frames,
                    total_frames
                ])

                print("  Finished:", filename)

print("Dataset saved to:", output_path)