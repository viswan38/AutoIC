import csv
import os

folder = "interaction_data"

for filename in os.listdir(folder):

    if not filename.endswith(".csv"):
        continue

    path = os.path.join(folder, filename)

    movements = []
    distances = []

    with open(path, "r") as file:

        reader = csv.DictReader(file)

        for row in reader:

            movements.append(float(row["chair_movement"]))
            distances.append(float(row["distance"]))

    if len(movements) == 0:
        continue

    average_movement = sum(movements) / len(movements)
    maximum_movement = max(movements)

    average_distance = sum(distances) / len(distances)

    print()
    print(filename)
    print("Average chair movement:", round(average_movement, 2))
    print("Maximum chair movement:", round(maximum_movement, 2))
    print("Average person-chair distance:", round(average_distance, 2))