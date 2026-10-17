import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("ucf11_dataset.csv")

# Remove videos where no person movement was measured
data = data[data["detected_frames"] > 1].copy()

# Select input features
X = data[[
    "average_movement",
    "maximum_movement",
    "movement_std",
    "average_width",
    "average_height",
    "detection_ratio",
    "total_frames"
]]

# Convert action names into numerical labels
encoder = LabelEncoder()
y = encoder.fit_transform(data["action"])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create neural network
model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    max_iter=500,
    random_state=42,
    early_stopping=True
)

# Train model
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print()
print(classification_report(
    y_test,
    predictions,
    labels=range(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
))