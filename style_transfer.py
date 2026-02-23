"""
style_transfer.py - Composition d'images ScrabbleGAN sur fond de document réel
===============================================================================

Prend des images synthétiques générées par ScrabbleGAN et les compose sur
des patches de fond extraits de vrais documents historiques, pour produire
des images plus réalistes visuellement et en couleur.

Workflow :
  1. Charger les images de fond depuis --real_dir (dossier "wall/" de patches curatés)
  2. Pour chaque image synthétique :
     a. Normaliser le synth (le GAN génère sur fond gris foncé, max ~106)
     b. Tirer un patch de fond aléatoire et le lisser légèrement
     c. Normaliser le fond globalement (préserve la teinte)
     d. Composer : fond réel là où c'est blanc, encre sombre là où c'est noir
  3. Sauvegarder en RGB

Usage :
    # Image unique
    python style_transfer.py \\
        --synth "synthetic_v2_ft/generated/00001_Histologie_000.png" \\
        --real_dir wall/ --output test_style.png

    # Dossier entier
    python style_transfer.py \\
        --synth_dir synthetic_v2_ft/generated/ \\
        --real_dir wall/ --output_dir synthetic_v2_styled/

    # Ajuster les paramètres
    python style_transfer.py \\
        --synth_dir synthetic_v2_ft/generated/ \\
        --real_dir wall/ --output_dir synthetic_v2_styled/ \\
        --blur 1.5 --bg_low 160 --bg_high 240
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


# =============================================================================
# 1. CHARGEMENT DES FONDS
# =============================================================================

def load_backgrounds(real_dir: str) -> list:
    """Charge toutes les images du dossier wall/ en RGB."""
    real_dir = Path(real_dir)
    imgs = []
    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        imgs.extend(real_dir.glob(f'*{ext}'))
    if not imgs:
        raise ValueError(f"Aucune image trouvée dans {real_dir}")
    backgrounds = []
    for p in imgs:
        try:
            backgrounds.append(Image.open(p).convert('RGB'))
        except Exception as e:
            print(f"  [!] {p.name} : {e}")
    print(f"  -> {len(backgrounds)} fonds chargés depuis {real_dir}")
    return backgrounds


# =============================================================================
# 2. COMPOSITION
# =============================================================================

def compose(synth_path: str, backgrounds: list,
            blur_radius: float = 1.5,
            bg_low: float = 160,
            bg_high: float = 240) -> Image.Image:
    """
    Compose une image synthétique ScrabbleGAN sur un fond réel en RGB.

    La normalisation du fond est globale (pas par canal) pour préserver
    la teinte naturelle du papier (beige, jaunâtre, etc.).

    Args:
        synth_path  : chemin vers l'image synthétique (grayscale)
        backgrounds : liste d'images PIL de fond RGB
        blur_radius : lissage gaussien du fond (0 = désactivé)
        bg_low      : borne basse de la normalisation du fond (défaut 160)
        bg_high     : borne haute de la normalisation du fond (défaut 240)

    Returns:
        Image PIL RGB composée
    """
    synth = Image.open(synth_path).convert('L')
    sw, sh = synth.size
    synth_arr = np.array(synth).astype(float)

    # Normaliser le synth vers 0-255 (le GAN génère souvent max ~106)
    s_min, s_max = synth_arr.min(), synth_arr.max()
    if s_max > s_min:
        synth_norm = (synth_arr - s_min) / (s_max - s_min) * 255
    else:
        synth_norm = synth_arr.copy()

    # Fond : tirage aléatoire + recadrage
    bg = random.choice(backgrounds)
    bw, bh = bg.size
    if bw >= sw and bh >= sh:
        x = random.randint(0, bw - sw)
        y = random.randint(0, bh - sh)
        bg = bg.crop((x, y, x + sw, y + sh))
    else:
        bg = bg.resize((sw, sh), Image.LANCZOS)

    # Lissage léger
    if blur_radius > 0:
        bg = bg.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    wall_arr = np.array(bg).astype(float)

    # Normalisation GLOBALE du fond (préserve la teinte entre canaux)
    wall_min = wall_arr.min()
    wall_max = wall_arr.max()
    wall_norm = (wall_arr - wall_min) / (wall_max - wall_min + 1e-6) * (bg_high - bg_low) + bg_low

    # Masque : 0=encre, 1=fond
    mask = synth_norm / 255.0                          # (h, w)
    mask3 = np.stack([mask, mask, mask], axis=2)       # (h, w, 3)

    # Encre : fond teinté très sombre
    ink = wall_norm * (synth_norm[:, :, np.newaxis] / 255.0) * 0.2

    # Composition
    result = mask3 * wall_norm + (1 - mask3) * ink
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result, mode='RGB')


# =============================================================================
# 3. PIPELINE PRINCIPAL
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Style transfer : images ScrabbleGAN sur fond de document réel (RGB)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Image unique
  python style_transfer.py \\
      --synth "synthetic_v2_ft/generated/00001_Histologie_000.png" \\
      --real_dir wall/ --output test_style.png

  # Dossier entier
  python style_transfer.py \\
      --synth_dir synthetic_v2_ft/generated/ \\
      --real_dir wall/ --output_dir synthetic_v2_styled/

  # Paramètres personnalisés
  python style_transfer.py \\
      --synth_dir synthetic_v2_ft/generated/ \\
      --real_dir wall/ --output_dir synthetic_v2_styled/ \\
      --blur 1.0 --bg_low 150 --bg_high 230
        """
    )
    parser.add_argument('--synth',      help='Image synthétique unique en entrée')
    parser.add_argument('--synth_dir',  help="Dossier d'images synthétiques")
    parser.add_argument('--real_dir',   required=True,
                        help='Dossier contenant les images de fond (wall/)')
    parser.add_argument('--output',     help='Image de sortie (mode fichier unique)')
    parser.add_argument('--output_dir', help='Dossier de sortie (mode dossier)')
    parser.add_argument('--blur',       type=float, default=1.5,
                        help='Rayon du flou gaussien sur le fond (défaut: 1.5, 0=désactivé)')
    parser.add_argument('--bg_low',     type=float, default=160,
                        help='Borne basse normalisation fond (défaut: 160)')
    parser.add_argument('--bg_high',    type=float, default=240,
                        help='Borne haute normalisation fond (défaut: 240)')
    parser.add_argument('--seed',       type=int, default=None,
                        help='Graine aléatoire pour la reproductibilité')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print("Chargement des fonds...")
    backgrounds = load_backgrounds(args.real_dir)

    if args.synth:
        output = args.output or str(Path(args.synth).stem) + '_styled.png'
        img = compose(args.synth, backgrounds, args.blur, args.bg_low, args.bg_high)
        img.save(output)
        print(f"-> {output}")

    elif args.synth_dir:
        synth_dir = Path(args.synth_dir)
        out_dir = Path(args.output_dir) if args.output_dir \
                  else synth_dir.parent / (synth_dir.name + '_styled')
        out_dir.mkdir(parents=True, exist_ok=True)

        synth_files = sorted(synth_dir.glob('*.png'))
        print(f"Traitement de {len(synth_files)} images...")

        for synth_path in synth_files:
            try:
                img = compose(str(synth_path), backgrounds, args.blur,
                              args.bg_low, args.bg_high)
                img.save(out_dir / synth_path.name)
            except Exception as e:
                print(f"  [!] {synth_path.name} : {e}")

        # Copier les transcriptions (.txt) dans le dossier de sortie
        for txt_path in synth_dir.glob('*.txt'):
            (out_dir / txt_path.name).write_text(
                txt_path.read_text(encoding='utf-8'), encoding='utf-8')

        print(f"-> {out_dir} ({len(synth_files)} images)")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
