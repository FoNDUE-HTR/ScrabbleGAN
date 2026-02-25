import os
import cv2
import numpy as np

# ===== CONFIG =====
INPUT_DIR = "data"
OUTPUT_DIR = "data_gamma"
EXTENSIONS = (".jpg", ".jpeg", ".png")
GAMMA = 0.8  # Ajustable

os.makedirs(OUTPUT_DIR, exist_ok=True)


def gamma_correction_gray(img, gamma=0.8):
    """
    Gamma correction sans modifier la géométrie.
    Conserve exactement les dimensions.
    """
    img = img.astype(np.float32) / 255.0
    img = np.power(img, gamma)
    img = (img * 255.0).clip(0, 255)
    return img.astype(np.uint8)


print("Processing images...")

for filename in os.listdir(INPUT_DIR):

    if not filename.lower().endswith(EXTENSIONS):
        continue

    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    corrected = gamma_correction_gray(img, gamma=GAMMA)

    cv2.imwrite(output_path, corrected)

print("Done.")