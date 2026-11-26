import os

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC

from features import extract_features, extract_features_from_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
ALLOWED = {".jpg", ".jpeg", ".png"}
CATEGORIES = ["biodegradable", "recyclable", "residual", "special"]


def source_group(path):
    filename = os.path.splitext(os.path.basename(path))[0]
    if filename.startswith("aug_"):
        filename = filename[4:]
        for suffix in ("_bright", "_dark", "_contrast", "_flip", "_rotate", "_blur", "_sharp", "_noise", "_zoom"):
            if filename.endswith(suffix):
                filename = filename[:-len(suffix)]
                break
    return filename


def is_clear_training_image(path):
    filename = os.path.splitext(os.path.basename(path))[0]
    return not filename.startswith("aug_")


def load_images():
    paths = []
    labels = []
    for category in CATEGORIES:
        folder = os.path.join(DATASET_DIR, category)
        for filename in os.listdir(folder):
            if os.path.splitext(filename)[1].lower() not in ALLOWED:
                continue
            path = os.path.join(folder, filename)
            if cv2.imread(path) is not None:
                paths.append(path)
                labels.append(category)
    features = np.array([extract_features(path) for path in paths])
    return paths, np.array(labels), features, np.array([source_group(path) for path in paths])


def augmented_training(paths, labels):
    features = []
    augmented_labels = []
    for path, label in zip(paths, labels):
        image = cv2.imread(path)
        variants = [
            image,
            cv2.flip(image, 1),
            cv2.convertScaleAbs(image, alpha=1.0, beta=20),
            cv2.convertScaleAbs(image, alpha=1.0, beta=-20),
        ]
        for variant in variants:
            features.append(extract_features_from_image(variant))
            augmented_labels.append(label)
    return np.array(features), np.array(augmented_labels)


paths, labels, features, groups = load_images()
accuracies = []
category_hits = {category: [] for category in CATEGORIES}

for round_number in range(10):
    splitter = StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=round_number
    )
    fold_accuracies = []
    fold_true = []
    fold_predicted = []

    for train_indices, test_indices in splitter.split(features, labels, groups):
        train_paths = [paths[index] for index in train_indices if is_clear_training_image(paths[index])]
        train_labels = [labels[index] for index in train_indices if is_clear_training_image(paths[index])]
        test_indices = [index for index in test_indices if not paths[index].split(os.sep)[-1].startswith("aug_")]
        if not test_indices:
            continue
        train_features, train_targets = augmented_training(train_paths, train_labels)
        model = SVC(
            C=3.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=round_number,
        )
        model.fit(train_features, train_targets)
        predictions = model.predict(features[test_indices])
        fold_accuracies.append(accuracy_score(labels[test_indices], predictions))
        fold_true.extend(labels[test_indices])
        fold_predicted.extend(predictions)

    accuracies.append(float(np.mean(fold_accuracies)))
    matrix = confusion_matrix(fold_true, fold_predicted, labels=CATEGORIES)
    for category_index, category in enumerate(CATEGORIES):
        total = matrix[category_index].sum()
        category_hits[category].append(
            float(matrix[category_index, category_index] / total) if total else 0.0
        )
    print(f"Round {round_number + 1}: accuracy={accuracies[-1]:.4f}")

print(f"Average accuracy: {np.mean(accuracies):.4f}")
print(f"Minimum accuracy: {np.min(accuracies):.4f}")
print(f"Maximum accuracy: {np.max(accuracies):.4f}")
for category in CATEGORIES:
    print(f"{category} average recall: {np.mean(category_hits[category]):.4f}")
