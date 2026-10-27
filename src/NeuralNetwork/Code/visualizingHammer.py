
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from collections import deque

video_path = Path("./CoreSet/Hammer nails/2BwhSTldTdg_0_00_0_10.mp4")
output_path = "hammer_visualization.mp4"

model = YOLO("yolo11n-pose.pt")
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise FileNotFoundError("Could not open the video: " + str(video_path))

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# COCO pose keypoint indexes
keypoint_names = {
    7: "Left elbow",
    8: "Right elbow",
    9: "Left wrist",
    10: "Right wrist",
    11: "Left hip",
    12: "Right hip"
}

trajectories = {}
previous_points = {}
movement_history = {
    "Wrist": [],
    "Elbow": [],
    "Hip": [],
    "Body": []
}

frame_number = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    results = model.predict(frame, verbose=False, conf=0.25)
    result = results[0]

    current_points = {}
    movement_values = {
        "Wrist": [],
        "Elbow": [],
        "Hip": [],
        "Body": []
    }

    if result.keypoints is not None and len(result.keypoints.xy) > 0:
        points = result.keypoints.xy[0].cpu().numpy()
        confidence = result.keypoints.conf[0].cpu().numpy()

        # Draw detected pose and skeleton
        annotated = result.plot()

        for index, name in keypoint_names.items():
            if confidence[index] < 0.25:
                continue

            x, y = points[index]
            current_points[index] = (int(x), int(y))

            if index not in trajectories:
                trajectories[index] = deque(maxlen=40)

            trajectories[index].append((int(x), int(y)))

            cv2.circle(annotated, (int(x), int(y)), 6, (0, 255, 255), -1)
            cv2.putText(
                annotated, name, (int(x) + 5, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2
            )
            cv2.putText(
                annotated, name, (int(x) + 5, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1
            )

            # Draw recent movement trajectory
            trail = list(trajectories[index])

            for i in range(1, len(trail)):
                cv2.line(
                    annotated, trail[i - 1], trail[i],
                    (255, 0, 255), 2
                )

            if index in previous_points:
                old_x, old_y = previous_points[index]
                distance = np.sqrt(
                    (x - old_x) ** 2 + (y - old_y) ** 2
                )

                if index in [9, 10]:
                    movement_values["Wrist"].append(distance)
                elif index in [7, 8]:
                    movement_values["Elbow"].append(distance)
                elif index in [11, 12]:
                    movement_values["Hip"].append(distance)

        previous_points = current_points.copy()

        # Average movement across detected body keypoints
        all_distances = []
        for index in current_points:
            if index in trajectories and len(trajectories[index]) >= 2:
                trail = trajectories[index]
                p1 = trail[-2]
                p2 = trail[-1]
                all_distances.append(
                    np.sqrt((p2[0] - p1[0]) ** 2 +
                            (p2[1] - p1[1]) ** 2)
                )

        movement_values["Body"] = all_distances

    else:
        annotated = frame.copy()

    # Save per-frame movement measurements
    for name in movement_history:
        values = movement_values[name]
        if len(values) > 0:
            movement_history[name].append(float(np.mean(values)))

    # Presentation information panel
    panel_height = 150
    cv2.rectangle(
        annotated, (0, 0), (width, panel_height), (25, 25, 25), -1
    )

    lines = [
        "AutoIC | Action: Hammer nails",
        "Frame: " + str(frame_number),
        "Yellow points: wrists / elbows / hips",
        "Purple trails: recent keypoint movement"
    ]

    for i, line in enumerate(lines):
        cv2.putText(
            annotated, line, (15, 28 + i * 27),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 255), 2
        )

    y_position = panel_height + 25

    for name in movement_history:
        values = movement_history[name]
        current = movement_values[name]

        current_value = np.mean(current) if current else 0
        average_value = np.mean(values) if values else 0

        text = (
            name + " movement (px/frame): "
            + str(round(current_value, 2))
            + " | Running avg: " + str(round(average_value, 2))
        )

        cv2.putText(
            annotated, text, (15, y_position),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 255), 2
        )

        y_position += 25

    writer.write(annotated)

cap.release()
writer.release()

print("Visualization saved to:", output_path)
