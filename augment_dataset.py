import os

import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
ALLOWED = {".jpg", ".jpeg", ".png"}


def create_variants(image):
    height, width = image.shape[:2]
    brighter = cv2.convertScaleAbs(image, alpha=1.0, beta=25)
    darker = cv2.convertScaleAbs(image, alpha=1.0, beta=-25)
    contrast = cv2.convertScaleAbs(image, alpha=1.2, beta=0)
    flipped = cv2.flip(image, 1)
    rotated = cv2.warpAffine(
        image,
        cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 8, 1.0),
        (image.shape[1], image.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    noise = np.random.default_rng(42).normal(0, 8, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    resized = cv2.resize(cv2.resize(image, None, fx=0.88, fy=0.88), (image.shape[1], image.shape[0]))
    rotate_left = cv2.warpAffine(
        image,
        cv2.getRotationMatrix2D((width / 2, height / 2), -8, 1.0),
        (width, height),
        borderMode=cv2.BORDER_REFLECT,
    )
    rotate_more = cv2.warpAffine(
        image,
        cv2.getRotationMatrix2D((width / 2, height / 2), 15, 1.0),
        (width, height),
        borderMode=cv2.BORDER_REFLECT,
    )
    vertical_flip = cv2.flip(image, 0)
    gray = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    saturated = cv2.convertScaleAbs(image, alpha=1.0, beta=0)
    saturated_hsv = cv2.cvtColor(saturated, cv2.COLOR_BGR2HSV)
    saturated_hsv[:, :, 1] = np.clip(saturated_hsv[:, :, 1].astype(np.int16) + 35, 0, 255)
    saturated = cv2.cvtColor(saturated_hsv, cv2.COLOR_HSV2BGR)
    muted_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    muted_hsv[:, :, 1] = (muted_hsv[:, :, 1] * 0.55).astype(np.uint8)
    muted = cv2.cvtColor(muted_hsv, cv2.COLOR_HSV2BGR)
    warm = image.copy()
    warm[:, :, 2] = np.clip(warm[:, :, 2].astype(np.int16) + 20, 0, 255).astype(np.uint8)
    cool = image.copy()
    cool[:, :, 0] = np.clip(cool[:, :, 0].astype(np.int16) + 20, 0, 255).astype(np.uint8)
    crop = image[int(height * 0.08):int(height * 0.92), int(width * 0.08):int(width * 0.92)]
    crop = cv2.resize(crop, (width, height))
    stretch = cv2.resize(image, (width, int(height * 1.12)))
    stretch = stretch[:height, :]
    shadow = cv2.convertScaleAbs(image, alpha=0.75, beta=-10)
    high_contrast = cv2.convertScaleAbs(image, alpha=1.45, beta=-35)
    low_contrast = cv2.convertScaleAbs(image, alpha=0.75, beta=30)
    tint = image.copy()
    tint[:, :, 1] = np.clip(tint[:, :, 1].astype(np.int16) + 15, 0, 255).astype(np.uint8)
    soft = cv2.GaussianBlur(image, (3, 3), 0)
    poster = cv2.convertScaleAbs(image, alpha=1.65, beta=-70)
    return {
        "bright": brighter,
        "dark": darker,
        "contrast": contrast,
        "flip": flipped,
        "rotate": rotated,
        "blur": blurred,
        "sharp": sharpened,
        "noise": noisy,
        "zoom": resized,
        "rotate_left": rotate_left,
        "rotate_more": rotate_more,
        "vertical_flip": vertical_flip,
        "gray": gray,
        "saturated": saturated,
        "muted": muted,
        "warm": warm,
        "cool": cool,
        "crop": crop,
        "stretch": stretch,
        "shadow": shadow,
        "high_contrast": high_contrast,
        "low_contrast": low_contrast,
        "tint": tint,
        "soft": soft,
        "poster": poster,
    }


def augment_category(category):
    folder = os.path.join(DATASET_DIR, category)
    source_files = [
        name for name in os.listdir(folder)
        if not name.startswith("aug_")
        and os.path.splitext(name)[1].lower() in ALLOWED
    ]
    created = 0

    for filename in source_files:
        source_path = os.path.join(folder, filename)
        image = cv2.imread(source_path)
        if image is None:
            print(f"Skipped unreadable image: {source_path}")
            continue

        stem = os.path.splitext(filename)[0]
        for variant_name, variant in create_variants(image).items():
            output_name = f"aug_{stem}_{variant_name}.jpg"
            output_path = os.path.join(folder, output_name)
            if os.path.exists(output_path):
                continue
            cv2.imwrite(output_path, variant, [cv2.IMWRITE_JPEG_QUALITY, 95])
            created += 1

    print(f"{category}: {len(source_files)} originals, {created} new variants")


if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError("Run this script from a project containing dataset/.")

categories = sorted(
    name for name in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, name))
)

for category in categories:
    augment_category(category)

print("Dataset augmentation complete. Run py train_model.py next.")
