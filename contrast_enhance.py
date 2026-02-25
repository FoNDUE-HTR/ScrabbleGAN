"""
binarisation.py - Gamma correction sur les images avant fine-tuning ScrabbleGAN
================================================================================

Applique une correction gamma aux images d'un dossier pour améliorer
le contraste avant l'entraînement du GAN.

  gamma < 1 : assombrit l'image (utile si le fond est trop clair)
  gamma > 1 : éclaircit l'image (utile si l'image est trop sombre)

Les fichiers XML (ALTO) associés sont copiés tels quels dans le dossier
de sortie pour conserver les paires image/transcription.

Usage :
    python binarisation.py --input_dir data/ --output_dir data_gamma/
    python binarisation.py --input_dir data/ --output_dir data_gamma/ --gamma 1.2
"""

import argparse
import shutil
import os
import cv2
import numpy as np
from pathlib import Path


EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    Gamma correction sans modifier la géométrie.
    gamma < 1 : assombrit, gamma > 1 : éclaircit.
    """
    img = img.astype(np.float32) / 255.0
    img = np.power(img, gamma)
    return (img * 255.0).clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description='Gamma correction sur les images avant fine-tuning ScrabbleGAN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Assombrir légèrement (gamma < 1)
  python binarisation.py --input_dir data/ --output_dir data_gamma/ --gamma 0.8

  # Éclaircir (gamma > 1)
  python binarisation.py --input_dir data/ --output_dir data_gamma/ --gamma 1.2
        """
    )
    parser.add_argument('--input_dir',  required=True, help='Dossier d\'images en entrée')
    parser.add_argument('--output_dir', required=True, help='Dossier de sortie')
    parser.add_argument('--gamma',      type=float, default=0.8,
                        help='Valeur gamma (défaut: 0.8, <1=assombrit, >1=éclaircit)')
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_files = [f for f in sorted(input_dir.iterdir())
                 if f.suffix.lower() in EXTENSIONS]
    xml_files = sorted(input_dir.glob("*.xml"))

    print(f"Gamma : {args.gamma}")
    print(f"Images : {len(img_files)}, XML : {len(xml_files)}")

    # Traiter les images
    ok, skipped = 0, 0
    for f in img_files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [!] Impossible de lire {f.name}")
            skipped += 1
            continue
        corrected = gamma_correction(img, args.gamma)
        cv2.imwrite(str(output_dir / f.name), corrected)
        ok += 1

    # Copier les XML tels quels
    for f in xml_files:
        shutil.copy(f, output_dir / f.name)

    print(f"-> {ok} images traitées, {len(xml_files)} XML copiés, {skipped} ignorées")
    print(f"-> {output_dir}")


if __name__ == '__main__':
    main()
