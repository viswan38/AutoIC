import cv2
import math
import csv
import os
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

folders = ["noPickupChair", "pickUpChair"]

os.makedirs("interaction_data", exist_ok=True)

for folder in folders:

    for filename in os.listdir(folder):

        if not filename.endswith(".mp4"):
            continue

        video_path = os.path.join(folder, filename)

        video = cv2.VideoCapture(video_path)

        print("Processing:", video_path)
        print("Opened:", video.isOpened())

        if not video.isOpened():
            print("Could not open:", video_path)
            continue

        data = []
        frame_number = 0

        previous_chair_x = None
        previous_chair_y = None

        while True:

            success, frame = video.read()

            if not success:
                break

            frame_number += 1

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0, 56]
            )

            people = []
            chairs = []

            for result in results:

                boxes = result.boxes

                for box in boxes:

                    x1, y1, x2, y2 = box.xyxy[0]

                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)

                    class_id = int(box.cls[0])

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    if class_id == 0:
                        people.append(
                            (x1, y1, x2, y2, center_x, center_y)
                        )

                    if class_id == 56:
                        chairs.append(
                            (x1, y1, x2, y2, center_x, center_y)
                        )

            if len(chairs) > 0 and len(people) > 0:

                chair = chairs[0]

                chair_x = chair[4]
                chair_y = chair[5]

                chair_center = (chair_x, chair_y)

                if previous_chair_x is None:

                    chair_movement = 0

                else:

                    chair_movement = math.sqrt(
                        (chair_x - previous_chair_x) ** 2 +
                        (chair_y - previous_chair_y) ** 2
                    )

                previous_chair_x = chair_x
                previous_chair_y = chair_y

                closest_person = None
                closest_distance = float("inf")

                for person in people:

                    person_center = (person[4], person[5])

                    distance = math.sqrt(
                        (person_center[0] - chair_center[0]) ** 2 +
                        (person_center[1] - chair_center[1]) ** 2
                    )

                    if distance < closest_distance:

                        closest_distance = distance
                        closest_person = person

                x1, y1, x2, y2, person_x, person_y = closest_person

                data.append([
                    frame_number,
                    person_x,
                    person_y,
                    chair_x,
                    chair_y,
                    closest_distance,
                    chair_movement,
                    folder
                ])

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.rectangle(
                    frame,
                    (chair[0], chair[1]),
                    (chair[2], chair[3]),
                    (0, 255, 0),
                    2
                )

                cv2.line(
                    frame,
                    (person_x, person_y),
                    chair_center,
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Distance: " + str(round(closest_distance)),
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Chair Movement: " + str(round(chair_movement, 2)),
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Person Chair Interaction", frame)

            if cv2.waitKey(1) == ord("q"):
                break

        video.release()

        output_name = os.path.splitext(filename)[0] + ".csv"

        output_path = os.path.join(
            "interaction_data",
            folder + "_" + output_name
        )

        with open(output_path, "w", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                "frame",
                "person_x",
                "person_y",
                "chair_x",
                "chair_y",
                "distance",
                "chair_movement",
                "label"
            ])

            writer.writerows(data)

        print("Saved:", output_path)

cv2.destroyAllWindows()

print("Finished processing all videos.")