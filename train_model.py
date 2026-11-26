import os
import pickle
import cv2
import numpy as np
from features import extract_features, extract_features_from_image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_FILE = os.path.join(MODEL_DIR, "cleanspot_model.pkl")
CLASSES_FILE = os.path.join(MODEL_DIR, "classes.txt")

ALLOWED = {".jpg", ".jpeg", ".png"}
REQUIRED_CATEGORIES = {"biodegradable", "recyclable", "residual", "special"}

def load_dataset():
    paths = []
    labels = []

    if not os.path.isdir(DATASET):
        raise FileNotFoundError("Create the dataset folder first.")

    classes = sorted(
        name for name in os.listdir(DATASET)
        if os.path.isdir(os.path.join(DATASET, name))
    )

    missing_categories = REQUIRED_CATEGORIES - set(classes)
    if missing_categories:
        missing = ", ".join(sorted(missing_categories))
        raise ValueError(f"Missing required category folder(s): {missing}")

    for label in classes:
        folder = os.path.join(DATASET, label)
        count = 0

        for filename in os.listdir(folder):
            if os.path.splitext(filename)[1].lower() not in ALLOWED:
                continue

            path = os.path.join(folder, filename)

            try:
                image = cv2.imread(path)
                if image is None:
                    raise ValueError("OpenCV could not read the image.")
                paths.append(path)
                labels.append(label)
                count += 1
            except Exception as error:
                print(f"Skipped {filename}: {error}")

        print(f"{label}: {count} images")

        if count == 0:
            raise ValueError(f"Add at least one usable image to dataset/{label}/.")

    if not paths:
        raise ValueError("No usable images were found in the dataset.")

    return paths, np.array(labels), classes

image_paths, y, classes = load_dataset()
X = np.array([extract_features(path) for path in image_paths])

def source_group(path):
    filename = os.path.splitext(os.path.basename(path))[0]
    if filename.startswith("aug_"):
        filename = filename[4:]
        for suffix in ("_bright", "_dark", "_contrast", "_flip"):
            if filename.endswith(suffix):
                filename = filename[:-len(suffix)]
                break
    return filename

def is_clear_training_image(path):
    filename = os.path.splitext(os.path.basename(path))[0]
    return not filename.startswith("aug_")

def is_webcam_image(path):
    return os.path.basename(path).startswith("webcam_")

groups = np.array([source_group(path) for path in image_paths])

print(f"Total images: {len(X)}")
print(f"Classes: {classes}")
webcam_count = sum(is_webcam_image(path) for path in image_paths)
print(f"Labeled webcam images included: {webcam_count}")

# A train/test split is only possible when every class has enough samples.
if len(X) >= len(classes) * 2:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_indices, test_indices = next(splitter.split(X, y, groups))
    test_indices = [
        index for index in test_indices
        if not os.path.basename(image_paths[index]).startswith("aug_")
    ]
    X_test, y_test = X[test_indices], y[test_indices]
else:
    train_indices = np.arange(len(X))
    X_test, y_test = X, y
    print("WARNING: Dataset is very small; test accuracy is not a reliable evaluation.")

def augmented_features(paths):
    augmented_X = []
    augmented_y = []

    for path in paths:
        image = cv2.imread(path)
        variants = [
            image,
            cv2.flip(image, 1),
            cv2.convertScaleAbs(image, alpha=1.0, beta=20),
            cv2.convertScaleAbs(image, alpha=1.0, beta=-20)
        ]
        for variant in variants:
            augmented_X.append(extract_features_from_image(variant))
            augmented_y.append(y[image_paths.index(path)])

    return np.array(augmented_X), np.array(augmented_y)

train_groups = set(groups[train_indices])
train_paths = [
    path for index, path in enumerate(image_paths)
    if is_clear_training_image(path) and groups[index] in train_groups
]
X_train, y_train = augmented_features(train_paths)

model = SVC(
    C=3.0,
    kernel="rbf",
    gamma="scale",
    class_weight="balanced",
    probability=True
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Test accuracy: {accuracy:.4f}")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Use augmented versions of every labelled image for the final application model.
clear_paths = [path for path in image_paths if is_clear_training_image(path)]
X_all, y_all = augmented_features(clear_paths)
model.fit(X_all, y_all)

os.makedirs(MODEL_DIR, exist_ok=True)

with open(MODEL_FILE, "wb") as file:
    pickle.dump(model, file)

with open(CLASSES_FILE, "w", encoding="utf-8") as file:
    for class_name in classes:
        file.write(class_name + "\n")

print(f"Model saved to {MODEL_FILE}")
