import cv2
import math
import csv
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video = cv2.VideoCapture("./no_pickup/IMG_1647.mp4")

print("Opened:", video.isOpened())

data = []
frame_number = 0

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

        chair_center = (chair[4], chair[5])

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
            chair_center[0],
            chair_center[1],
            closest_distance
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

    cv2.imshow("Person Chair Interaction", frame)

    if cv2.waitKey(1) == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

with open("chair_interaction_data.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "frame",
        "person_x",
        "person_y",
        "chair_x",
        "chair_y",
        "distance"
    ])

    writer.writerows(data)

print("Saved chair_interaction_data.csv")
