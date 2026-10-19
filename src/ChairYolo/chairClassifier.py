import csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = []

with open("chair_dataset.csv", "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        data.append([
            float(row["average_movement"]),
            float(row["maximum_movement"]),
            float(row["average_distance"]),
            float(row["average_person_movement"]),
            float(row["maximum_person_movement"]),
            float(row["moving_frames"]),
            float(row["movement_ratio"]),
            int(row["label"])
        ])

X = []
y = []

for row in data:

    X.append(row[:7])
    y.append(row[7])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

print("Actual:", y_test)
print("Predicted:", predictions)