
import csv
import os

folder = "interaction_data"
output_file = "chair_dataset.csv"
movement_threshold = 5

with open(output_file, "w", newline="") as file:

    writer = csv.writer(file)

    print("Writing:", output_file)

    writer.writerow([
        "video",
        "average_movement",
        "maximum_movement",
        "average_distance",
        "average_person_movement",
        "maximum_person_movement",
        "moving_frames",
        "movement_ratio",
        "label"
    ])

    for filename in os.listdir(folder):

        if not filename.endswith(".csv"):
            continue

        path = os.path.join(folder, filename)

        movements = []
        distances = []
        person_movements = []

        with open(path, "r") as input_file:

            reader = csv.DictReader(input_file)

            for row in reader:
                movements.append(float(row["chair_movement"]))
                distances.append(float(row["distance"]))
                person_movements.append(float(row["person_movement"]))

        if len(movements) == 0:
            continue

        average_movement = sum(movements) / len(movements)
        maximum_movement = max(movements)
        average_distance = sum(distances) / len(distances)

        average_person_movement = (
            sum(person_movements) / len(person_movements)
        )
        maximum_person_movement = max(person_movements)

        moving_frames = 0

        for movement in movements:
            if movement > movement_threshold:
                moving_frames += 1

        movement_ratio = moving_frames / len(movements)

        if "pickUpChair" in filename:
            label = 1
        else:
            label = 0

        print("Video:", filename)
        print("Moving frames:", moving_frames)
        print("Movement ratio:", movement_ratio)

        writer.writerow([
            filename,
            average_movement,
            maximum_movement,
            average_distance,
            average_person_movement,
            maximum_person_movement,
            moving_frames,
            movement_ratio,
            label
        ])

print("Saved:", output_file)
