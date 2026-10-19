import csv
import os
import matplotlib.pyplot as plt

folder = "interaction_data"

for filename in os.listdir(folder):

    if not filename.endswith(".csv"):
        continue

    path = os.path.join(folder, filename)

    frames = []
    movements = []

    with open(path, "r") as file:

        reader = csv.DictReader(file)

        for row in reader:

            frames.append(int(row["frame"]))
            movements.append(float(row["chair_movement"]))

    if len(frames) == 0:
        continue

    plt.figure()

    plt.plot(frames, movements)

    plt.xlabel("Frame")
    plt.ylabel("Chair Movement")

    plt.title(filename)

    plt.show()