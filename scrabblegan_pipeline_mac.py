"""
htr_synth_v2.py - Pipeline ScrabbleGAN (arshjot) avec poids RIMES + fine-tuning
=================================================================================

Utilise le repo arshjot/ScrabbleGAN (deja clone et patche) avec les poids RIMES.
Prerequis : voir README.md pour la conversion des poids et les patches manuels.

Workflow :
  1. finetune : fine-tune sur vos patches ALTO via train.py
  2. generate : genere des images synthetiques a partir de vos textes ALTO

Usage :
    python htr_synth_v2.py --step finetune \\
        --weights ./data_final.pt \\
        --charmap ./RIMES_char_map.pkl \\
        --alto_dir ./data \\
        --output_dir ./synthetic_v2 \\
        --epochs 100

    python htr_synth_v2.py --step generate \\
        --weights scrabblegan_arshjot/weights/model_best.pth.tar \\
        --charmap ./RIMES_char_map.pkl \\
        --alto_dir ./data \\
        --output_dir ./synthetic_v2_ft \\
        --n_images 5
"""

import sys
import argparse
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import subprocess
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

SCRABBLEGAN_REPO = "https://github.com/arshjot/ScrabbleGAN.git"
SCRABBLEGAN_DIR  = Path("./scrabblegan_arshjot")


# =============================================================================
# UTILITAIRES
# =============================================================================

def parse_alto(alto_path: str) -> list[dict]:
    tree = ET.parse(alto_path)
    root = tree.getroot()
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    lines = []
    for tl in root.iter(f"{ns}TextLine"):
        from html import unescape
        text = unescape(" ".join(s.get("CONTENT", "") for s in tl.iter(f"{ns}String")).strip())
        if text:
            lines.append({"id": tl.get("ID", ""), "text": text})
    return lines


def extract_texts(alto_dir: str) -> list[str]:
    """Retourne la liste de tous les textes uniques depuis les ALTO."""
    texts = set()
    for xml in sorted(Path(alto_dir).glob("**/*.xml")):
        for line in parse_alto(str(xml)):
            texts.add(line["text"])
    return sorted(texts)


def find_pairs(data_dir: str) -> list[tuple[str, str]]:
    pairs = []
    for xml in sorted(Path(data_dir).glob("**/*.xml")):
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            img = xml.with_suffix(ext)
            if img.exists():
                pairs.append((str(xml), str(img)))
                break
    return pairs


# =============================================================================
# 1. SETUP
# =============================================================================


# =============================================================================
# 2. GENERATION — utilise ImgGenerator directement (pas generate_images.py)
# =============================================================================

def step_generate(weights: str, charmap: str, alto_dir: str, output_dir: str, n_images: int):
    import pickle as _pkl
    import numpy as np
    from PIL import Image as PILImage

    print("=== GENERATION ===")

    # Ajouter le repo au path pour importer ImgGenerator
    if str(SCRABBLEGAN_DIR.resolve()) not in sys.path:
        sys.path.insert(0, str(SCRABBLEGAN_DIR.resolve()))

    # Extraire les textes depuis les ALTO
    print("[1/3] Extraction des textes depuis les fichiers ALTO...")
    texts = extract_texts(alto_dir)
    if not texts:
        print(f"[!] Aucun texte trouve dans {alto_dir}")
        return

    output_dir = Path(output_dir)
    gen_dir    = output_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Charger le char_map
    print("[2/3] Chargement du modele...")
    with open(charmap, "rb") as _f:
        char_map = _pkl.load(_f, encoding="latin1")
    valid_chars = set(char_map.keys()) if isinstance(char_map, dict) else set(char_map)

    # Filtrer les textes
    def clean(t):
        return "".join(c for c in t if c in valid_chars).strip()
    texts_clean = [clean(t) for t in texts]
    texts_clean = [(orig, cleaned) for orig, cleaned in zip(texts, texts_clean) if cleaned]
    print(f"  -> {len(texts_clean)} textes valides sur {len(texts)}")

    # Patcher config.py pour le lexique avec chemin absolu
    config_path = SCRABBLEGAN_DIR / "config.py"
    config_txt  = config_path.read_text(encoding="utf-8")
    lex_abs     = str((SCRABBLEGAN_DIR / "data" / "Lexicon" / "Lexique383.tsv").resolve())
    import re as _re
    num_chars_val = len(char_map) if isinstance(char_map, dict) else len(set(char_map))
    config_patched = _re.sub(
        r"lexicon_file\s*=.*",
        f"lexicon_file = r'{lex_abs}'",
        config_txt
    )
    config_patched = _re.sub(
        r"num_chars\s*=.*",
        f"num_chars = {num_chars_val}",
        config_patched
    )
    config_path.write_text(config_patched, encoding="utf-8")

    # Instancier ImgGenerator
    try:
        # Recharger config avec le bon chemin
        import importlib, sys as _sys
        for mod in list(_sys.modules.keys()):
            if mod in ("config", "generate_images"):
                del _sys.modules[mod]
        from generate_images import ImgGenerator
        from config import Config
        Config.num_chars = len(char_map) if isinstance(char_map, dict) else len(set(char_map))
        generator = ImgGenerator(
            checkpt_path=str(Path(weights).resolve()),
            config=Config,
            char_map=char_map
        )
    except Exception as e:
        print(f"[!] Impossible de charger ImgGenerator : {e}")
        return

    # Generer et sauvegarder directement
    print(f"[3/3] Generation de {len(texts_clean) * n_images} images...")
    manifest = []
    failed   = 0

    # Couleur de fond cible : beige/jaune typique des documents historiques
    # On echantillonne depuis les vraies images pour matcher la couleur
    bg_color = None
    _pairs = find_pairs(alto_dir)
    _samples = []
    for _, img_path in _pairs[:10]:
        try:
            sample = PILImage.open(img_path).convert("RGB")
            arr_s  = np.array(sample)
            _samples.append(tuple(int(np.percentile(arr_s[:,:,c], 95)) for c in range(3)))
        except Exception:
            pass
    if _samples:
        bg_color = tuple(int(np.mean([s[c] for s in _samples])) for c in range(3))
    if bg_color is None:
        bg_color = (220, 210, 185)  # beige par defaut
    print(f"  bg_color calculé sur {len(_samples)} images : {bg_color}")

    def normalize_img(img_arr, bg=bg_color):
        """Normalise vers [0,255] et applique la couleur de fond des originaux."""
        arr = np.array(img_arr, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        gray = (arr * 255).astype(np.uint8)
        # Convertir en RGB en tintant : blanc -> bg_color, noir -> noir
        rgb = np.stack([
            (gray.astype(np.float32) / 255 * bg[0]).astype(np.uint8),
            (gray.astype(np.float32) / 255 * bg[1]).astype(np.uint8),
            (gray.astype(np.float32) / 255 * bg[2]).astype(np.uint8),
        ], axis=2)
        return rgb

    def generate_line_image(generator, text, char_map, n_variants):
        """
        Genere n_variants images pour un texte pouvant contenir des espaces.
        Retourne une liste de (img_array, word_positions)
        word_positions = [(word_text, x1, x2), ...]
        """
        words = text.split()
        words_clean = []
        for w in words:
            cleaned = "".join(c for c in w if c in char_map)
            if cleaned:
                words_clean.append((w, cleaned))
        if not words_clean:
            return []

        results = []
        for _ in range(n_variants):
            word_imgs = []
            for orig_w, clean_w in words_clean:
                try:
                    imgs, _, _ = generator.generate(
                        random_num_imgs=1,
                        word_list=[clean_w]
                    )
                    word_imgs.append((orig_w, normalize_img(imgs[0])))
                except Exception:
                    pass
            if not word_imgs:
                continue

            # Assembler avec fondu entre les mots (cross-fade adaptatif)
            def blend_join(img_a, img_b, space=16):
                """Espace entre mots : couleur moyenne entre les deux bords + fondu."""
                h = img_a.shape[0]
                # Couleur moyenne entre bord droit de A et bord gauche de B
                col_a = img_a[:, -4:, :].astype(np.float32).mean(axis=(0,1))
                col_b = img_b[:, :4, :].astype(np.float32).mean(axis=(0,1))
                mid_color = ((col_a + col_b) / 2).astype(np.uint8)
                gap = np.stack([
                    np.full((h, space), mid_color[0], dtype=np.uint8),
                    np.full((h, space), mid_color[1], dtype=np.uint8),
                    np.full((h, space), mid_color[2], dtype=np.uint8),
                ], axis=2)
                # Fondu bord droit de A -> espace (seulement sur le fond, pas les lettres)
                fade = min(6, img_a.shape[1] // 2, space // 2)
                a = img_a.astype(np.float32)
                g = gap.astype(np.float32)
                w = np.linspace(1, 0, fade)[np.newaxis, :, np.newaxis]
                mask_a = (a[:, -fade:].mean(axis=2) > 120)[:, :, np.newaxis]
                blended_a = a[:, -fade:] * w + g[:, :fade] * (1 - w)
                g[:, :fade] = np.where(mask_a, blended_a, g[:, :fade])
                # Fondu espace -> bord gauche de B (seulement sur le fond)
                b = img_b.astype(np.float32)
                w2 = np.linspace(0, 1, fade)[np.newaxis, :, np.newaxis]
                mask_b = (b[:, :fade].mean(axis=2) > 120)[:, :, np.newaxis]
                blended_b = g[:, -fade:] * (1 - w2) + b[:, :fade] * w2
                g[:, -fade:] = np.where(mask_b, blended_b, g[:, -fade:])
                return np.concatenate([img_a, g.astype(np.uint8), img_b], axis=1)

            # Tracker les positions de chaque mot
            word_positions = []
            x_cursor = 0
            line_img = None
            for i, (orig_w, wimg) in enumerate(word_imgs):
                w_width = wimg.shape[1]
                if i == 0:
                    line_img = wimg
                    word_positions.append((orig_w, 0, w_width))
                    x_cursor = w_width
                else:
                    x1 = x_cursor + 16  # apres l espace de 16px
                    line_img = blend_join(line_img, wimg)
                    x2 = x1 + w_width
                    word_positions.append((orig_w, x1, x2))
                    x_cursor = x2

            results.append((line_img, word_positions))
        return results

    def build_alto(img_path: Path, orig_text: str, word_positions: list,
                   img_w: int, img_h: int, line_idx: int) -> str:
        """Genere un fichier ALTO v4 pour une image de ligne synthetique."""
        line_id   = f"eSc_line_{line_idx:06d}"
        block_id  = f"eSc_block_{line_idx:06d}"
        baseline  = f"0 {int(img_h * 0.80)} {img_w} {int(img_h * 0.80)}"
        poly_line = f"0 0 0 {img_h} {img_w} {img_h} {img_w} 0"

        strings_xml = []
        for w_idx, (word, x1, x2) in enumerate(word_positions):
            w_width = max(1, x2 - x1)
            # Glyphes : position proportionnelle dans le mot
            glyphs_xml = []
            for c_idx, char in enumerate(word):
                char_x = x1 + int(w_width * c_idx / max(1, len(word)))
                char_w = max(1, w_width // max(1, len(word)))
                glyph_poly = f"{char_x} 0 {char_x} {img_h} {char_x + char_w} {img_h} {char_x + char_w} 0"
                glyphs_xml.append(f"""              <Glyph ID="char_{w_idx}_{c_idx}"
                     CONTENT="{char}"
                     HPOS="{char_x}"
                     VPOS="0"
                     WIDTH="{char_w}"
                     HEIGHT="{img_h}"
                     GC="1.0000">
                <Shape><Polygon POINTS="{glyph_poly}"/></Shape>
              </Glyph>""")

            strings_xml.append(f"""            <String ID="segment_{w_idx}"
                    CONTENT="{word}"
                    HPOS="{x1}"
                    VPOS="0"
                    WIDTH="{w_width}"
                    HEIGHT="{img_h}"
                    WC="1.0000">
{''.join(glyphs_xml)}
            </String>""")

        return f"""<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xmlns="http://www.loc.gov/standards/alto/ns-v4#"
      xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v4# http://www.loc.gov/standards/alto/v4/alto-4-2.xsd">
  <Description>
    <MeasurementUnit>pixel</MeasurementUnit>
    <sourceImageInformation>
      <fileName>{img_path.name}</fileName>
    </sourceImageInformation>
  </Description>
  <Layout>
    <Page WIDTH="{img_w}"
          HEIGHT="{img_h}"
          PHYSICAL_IMG_NR="{line_idx}"
          ID="eSc_dummypage_">
      <PrintSpace HPOS="0" VPOS="0" WIDTH="{img_w}" HEIGHT="{img_h}">
        <TextBlock ID="{block_id}">
          <TextLine ID="{line_id}"
                    TAGREFS="LT1047"
                    BASELINE="{baseline}"
                    HPOS="0"
                    VPOS="0"
                    WIDTH="{img_w}"
                    HEIGHT="{img_h}">
            <Shape><Polygon POINTS="{poly_line}"/></Shape>
{''.join(strings_xml)}
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>"""

    for idx, (orig_text, clean_text) in enumerate(texts_clean):
        try:
            results = generate_line_image(generator, orig_text, char_map, n_images)
            safe = orig_text[:50].replace(" ", "_").replace("/", "-")
            for i, (arr, word_positions) in enumerate(results):
                img_path = gen_dir / f"{idx:05d}_{safe}_{i:03d}.png"
                txt_path = img_path.with_suffix(".txt")
                xml_path = img_path.with_suffix(".xml")
                from PIL import ImageFilter
                pil_img = PILImage.fromarray(arr)
                if False:  # --sharpen desactive
                    pil_img = pil_img.filter(ImageFilter.SHARPEN)
                    pil_img = pil_img.filter(ImageFilter.SHARPEN)
                    w, h = pil_img.size
                    pil_img = pil_img.resize((w * 2, h * 2), PILImage.LANCZOS)
                pil_img.save(img_path)
                img_w, img_h = pil_img.size
                # Ajuster positions si upscale x2
                scale = 1  # sharpen desactive
                scaled_positions = [(w, x1 * scale, x2 * scale) for w, x1, x2 in word_positions]
                txt_path.write_text(orig_text, encoding="utf-8")
                alto_xml = build_alto(img_path, orig_text, scaled_positions, img_w, img_h, idx * n_images + i)
                xml_path.write_text(alto_xml, encoding="utf-8")
                manifest.append((str(img_path), orig_text))
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  [!] Echec '{orig_text[:30]}' : {e}")

        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(texts_clean)} traites ({len(manifest)} images)...")

    # Manifeste
    manifest_path = output_dir / "manifest_generated.csv"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("image_path\ttext\n")
        for p, t in manifest:
            f.write(f"{p}\t{t}\n")

    # Restaurer config.py original
    config_path.write_text(config_txt, encoding="utf-8")

    print(f"\n[OK] {len(manifest)} images -> {gen_dir}")
    if failed:
        print(f"  ({failed} textes ont echoue)")
    print(f"  Manifeste : {manifest_path}")


def _create_txt_pairs(gen_dir: Path, texts: list[str]):
    """
    generate_images.py du repo arshjot nomme les images avec le texte dans le nom.
    Format : <text>_<idx>.png  ou  <idx>_<text>.png
    On cree le .txt correspondant.
    """
    # Construire un index texte -> set pour matching rapide
    text_set = set(texts)
    count = 0
    for img in sorted(gen_dir.glob("*.png")):
        stem  = img.stem
        # Essai : texte avant le dernier "_"
        parts = stem.rsplit("_", 1)
        text  = parts[0].replace("_", " ") if len(parts) == 2 and parts[1].isdigit() else stem
        img.with_suffix(".txt").write_text(text, encoding="utf-8")
        count += 1
    print(f"  -> {count} fichiers .txt crees")


def _write_manifest(gen_dir: Path, manifest_path: Path):
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("image_path\ttext\n")
        for img in sorted(gen_dir.glob("*.png")):
            txt  = img.with_suffix(".txt")
            text = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
            f.write(f"{img}\t{text}\n")
    print(f"  -> Manifeste : {manifest_path}")


# =============================================================================
# 3. FINE-TUNING via train.py du repo arshjot
# =============================================================================

def step_finetune(weights: str, charmap: str, alto_dir: str, output_dir: str, epochs: int, save_patches_dir=None):
    """
    Fine-tune ScrabbleGAN sur vos patches ALTO.

    Le repo arshjot utilise des fichiers pickle (pas LMDB comme amzn).
    Ce script :
      1. Extrait vos patches et les convertit au format attendu par train.py
      2. Lance train.py avec --continue_train
    """
    import pickle
    import numpy as np
    from PIL import Image

    print("=== FINE-TUNING ===")

    output_dir = Path(output_dir)
    ft_dir     = output_dir / "finetuned"
    ft_dir.mkdir(parents=True, exist_ok=True)

    # Charger le character mapping
    print(f"[1/4] Chargement du character mapping : {charmap}")
    with open(charmap, "rb") as f:
        char_map = pickle.load(f, encoding="latin1")
    if isinstance(char_map, dict):
        char2idx = char_map
    else:
        char2idx = {c: i for i, c in enumerate(char_map)}
    print(f"  -> {len(char2idx)} caracteres")

    # Extraction des patches depuis les ALTO
    print("[2/4] Extraction des patches ALTO...")
    if save_patches_dir:
        save_patches_dir = Path(save_patches_dir)
        save_patches_dir.mkdir(parents=True, exist_ok=True)
        print(f"  -> Patches sauvegardes dans {save_patches_dir}")
    pairs = find_pairs(alto_dir)
    if not pairs:
        print(f"[!] Aucune paire ALTO/image dans {alto_dir}")
        return

    samples = []  # [(img_array H=32, text), ...]
    for xml_path, img_path in pairs:

        try:
            img_full = Image.open(img_path).convert("L")
            w_img, h_img = img_full.size

            tree = ET.parse(xml_path)
            root = tree.getroot()
            ns   = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""

        except ET.ParseError as e:
            print(f"\n❌ XML invalide: {xml_path}")
            print(f"   Ligne: {e.position[0]}, Colonne: {e.position[1]}")
            print(f"   Message: {e}")
            continue  # ou raise si tu veux stopper

        # Extraction au niveau MOT (<String>) et non ligne (<TextLine>)
        # ScrabbleGAN est concu pour des mots isoles, pas des lignes entieres
        for s in root.iter(f"{ns}String"):
            from html import unescape
            text = unescape(s.get("CONTENT", "").strip())
            if not text:
                continue
            # Nettoyer le mot : garder seulement les caracteres dans l'alphabet
            text = "".join(c for c in text if c in char2idx).strip()
            if not text:
                continue
            hpos   = int(float(s.get("HPOS",   0)))
            vpos   = int(float(s.get("VPOS",   0)))
            width  = int(float(s.get("WIDTH",  0)))
            height = int(float(s.get("HEIGHT", 0)))
            if width <= 0 or height <= 0:
                continue
            x1, y1 = max(0, hpos), max(0, vpos)
            x2, y2 = min(w_img, hpos + width), min(h_img, vpos + height)
            if x2 <= x1 or y2 <= y1:
                continue
            patch = img_full.crop((x1, y1, x2, y2))
            ratio = 32 / patch.height
            new_w = max(1, int(patch.width * ratio))
            patch = patch.resize((new_w, 32), Image.LANCZOS)
            samples.append((np.array(patch), text))
            if save_patches_dir:
                safe = text[:30].replace(" ", "_").replace("/", "-")
                idx_str = f"{len(samples):05d}"
                patch_path = save_patches_dir / f"{idx_str}_{safe}.png"
                txt_path   = patch_path.with_suffix(".txt")
                patch.save(patch_path)
                txt_path.write_text(text, encoding="utf-8")

    print(f"  -> {len(samples)} patches extraits")

    # Sauvegarder au format EXACT attendu par data_generator.py :
    # { 'word_data': { id: (label_encoded, img_array_H32) }, 'char_map': {...} }
    print("[3/4] Preparation du dataset pickle (format word_data)...")
    data_pkl  = ft_dir / "custom_data.pkl"
    word_data = {}
    skipped   = 0
    for i, (arr, text) in enumerate(samples):
        label = [char2idx[c] for c in text if c in char2idx]
        if not label:
            skipped += 1
            continue
        word_data[i] = (label, arr)

    dataset = {"word_data": word_data, "char_map": char2idx}
    with open(data_pkl, "wb") as f:
        pickle.dump(dataset, f)
    print(f"  -> {len(word_data)} echantillons ({skipped} ignores) -> {data_pkl}")

    # Copier data_final.pt comme checkpoint de depart
    ckpt_copy = ft_dir / "checkpoint_finetune.pt"
    if not ckpt_copy.exists():
        shutil.copy(weights, ckpt_copy)
        print(f"  -> Checkpoint copie -> {ckpt_copy}")

    # Réécrire config.py proprement avec les bons chemins absolus
    config_path   = SCRABBLEGAN_DIR / "config.py"
    config_backup = SCRABBLEGAN_DIR / "config_backup.py"
    # Restaurer depuis backup si existe
    if config_backup.exists():
        original_config = config_backup.read_text(encoding="utf-8")
    else:
        original_config = config_path.read_text(encoding="utf-8")
        config_backup.write_text(original_config, encoding="utf-8")

    data_pkl_abs  = str(data_pkl.resolve())
    ckpt_dir_abs  = str(ckpt_copy.parent.resolve())
    lexicon_abs   = str((SCRABBLEGAN_DIR / 'data' / 'Lexicon' / 'Lexique383.tsv').resolve())
    ckpt_name     = ckpt_copy.name

    # Construire un config.py patch minimal
    # On surcharge uniquement les champs necessaires apres la classe
    patch = f"""
# === PATCH FINETUNE (ajoute automatiquement) ===
import torch as _torch

class Config:
    dataset = 'RIMES'
    data_folder_path = './RIMES/'
    img_h = 32
    char_w = 16
    partition = 'tr'
    batch_size = 8
    num_epochs = {epochs}
    epochs_lr_decay = {epochs}
    resume_training = True
    start_epoch = 0
    train_gen_steps = 4
    grad_alpha = 1
    grad_balance = True
    data_file = r'{data_pkl_abs}'
    lexicon_file_name = 'Lexique383.tsv'
    lexicon_file = r'{lexicon_abs}'
    lmdb_output = './data/custom_lmdb'
    architecture = 'ScrabbleGAN'
    r_ks = [3, 3, 3, 3, 3, 3, 2]
    r_pads = [1, 1, 1, 1, 1, 1, 0]
    r_fs = [64, 128, 256, 256, 512, 512, 512]
    resolution = 16
    bn_linear = 'SN'
    g_shared = False
    g_lr = 2e-4
    d_lr = 2e-4
    r_lr = 2e-4
    g_betas = [0., 0.999]
    d_betas = [0., 0.999]
    r_betas = [0., 0.999]
    g_loss_fn = 'HingeLoss'
    d_loss_fn = 'HingeLoss'
    r_loss_fn = 'CTCLoss'
    z_dim = 128
    num_chars = {len(char2idx)}
    weight_dir = r'{ckpt_dir_abs}'
    # Détection du device disponible
    if _torch.cuda.is_available():
        device = _torch.device('cuda')
        print('\033[32m[GPU] CUDA détecté et utilisé\033[0m')
    elif _torch.backends.mps.is_available():
        device = _torch.device('mps')
        print('\033[32m[GPU] MPS (Apple Silicon) détecté et utilisé\033[0m')
    else:
        device = _torch.device('cpu')
        print('\033[31m[CPU] Aucun GPU détecté - entraînement sur CPU\033[0m')
"""
    config_path.write_text(patch, encoding="utf-8")
    print(f"  -> config.py réécrit avec chemins absolus")

    # ModelCheckpoint cherche dans ./weights/ relatif au cwd (scrabblegan_arshjot/)
    weights_dir = SCRABBLEGAN_DIR.resolve() / "weights"
    weights_dir.mkdir(exist_ok=True)
    # Nettoyer les anciens checkpoints pour éviter les conflits d'architecture
    for old_ckpt in weights_dir.glob("*.pth.tar"):
        old_ckpt.unlink()
        print(f"  -> Ancien checkpoint supprime : {old_ckpt.name}")
    ckpt_in_weightdir = weights_dir / "model_checkpoint_epoch_0.pth.tar"
    shutil.copy(weights, ckpt_in_weightdir)
    print(f"  -> Checkpoint copie -> {ckpt_in_weightdir}")

    # Lancement de train.py depuis son repertoire
    print(f"[4/4] Lancement de train.py ({epochs} epochs)...")
    train_script = SCRABBLEGAN_DIR.resolve() / "train.py"
    cmd = [sys.executable, str(train_script)]
    print(f"  Commande : {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(SCRABBLEGAN_DIR.resolve()))

    # Restaurer config.py original apres l'entrainement
    config_path.write_text(original_config, encoding="utf-8")
    print(f"  -> config.py restaure")

    if result.returncode == 0:
        print(f"\n[OK] Fine-tuning termine -> {ft_dir}")
        print(f"\n  Regenerez :")
        print(f"  python htr_synth_v2.py --step generate --weights {ckpt_in_weightdir} ...")
    else:
        print(f"\n[!] train.py a retourne une erreur (voir ci-dessus)")
        print(f"    config.py restaure dans {config_path}")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ScrabbleGAN RIMES (arshjot) + fine-tuning sur vos ALTO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--step",       required=True,
                        choices=["generate", "finetune"])
    parser.add_argument("--weights",    default="./data.pkl",
                        help="Checkpoint ScrabbleGAN RIMES (data.pkl)")
    parser.add_argument("--charmap",    default="./char_map.pkl",
                        help="Character mapping RIMES (.pkl)")
    parser.add_argument("--alto_dir",   default="./data",
                        help="Dossier des paires ALTO/image")
    parser.add_argument("--output_dir", default="./synthetic_v2",
                        help="Dossier de sortie")
    parser.add_argument("--n_images",   type=int, default=5,
                        help="Images generees par texte ALTO")
    parser.add_argument("--save_patches", default=None, metavar="DIR",
                        help="Sauvegarder les patches image+txt dans ce dossier (optionnel)")
    parser.add_argument("--epochs",     type=int, default=10,
                        help="Epochs de fine-tuning")
    args = parser.parse_args()

    if args.step == "generate":
        step_generate(args.weights, args.charmap, args.alto_dir,
                      args.output_dir, args.n_images)
    elif args.step == "finetune":
        step_finetune(args.weights, args.charmap, args.alto_dir,
                      args.output_dir, args.epochs,
                      save_patches_dir=args.save_patches)


if __name__ == "__main__":
    main()
