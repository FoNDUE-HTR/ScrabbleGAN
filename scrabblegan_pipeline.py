"""
scrabblegan_pipeline.py - Pipeline ScrabbleGAN pour l'HTR français historique
==============================================================================

Pipeline unifié pour la génération de données synthétiques d'écriture manuscrite
à partir de ScrabbleGAN (arshjot) fine-tuné sur vos documents ALTO/image.

Étapes disponibles (dans l'ordre recommandé) :
  contrast   : (optionnel) correction gamma sur les images avant entraînement
  wordlevel  : conversion ALTO ligne-level -> word-level via Kraken
  normalize  : normalisation du texte ALTO selon un char_map (IAM/RIMES)
  finetune   : fine-tuning de ScrabbleGAN sur vos patches ALTO
  generate   : génération d'images synthétiques + ALTO v4
  style      : (optionnel) application de couleurs de documents réels

Usage :
    python scrabblegan_pipeline.py --step contrast \\
        --input_dir data/ --output_dir data_gamma/ --gamma 0.8

    python scrabblegan_pipeline.py --step wordlevel \\
        --xml_dir data/ --model models/fondue_archimed_v4.mlmodel \\
        --output_dir data_wordlevel/

    python scrabblegan_pipeline.py --step normalize \\
        --xml_dir data_wordlevel/ --charmap models/RIMES_char_map.pkl \\
        --output_dir data_normalized/

    python scrabblegan_pipeline.py --step finetune \\
        --weights models/RIMES_data_reshaped.pt \\
        --charmap models/RIMES_char_map.pkl \\
        --alto_dir data_normalized/ --output_dir synthetic/ --epochs 100

    python scrabblegan_pipeline.py --step generate \\
        --weights scrabblegan_arshjot/weights/model_best.pth.tar \\
        --charmap models/RIMES_char_map.pkl \\
        --alto_dir data_normalized/ --output_dir synthetic/ --n_images 5

    python scrabblegan_pipeline.py --step style \\
        --synth_dir synthetic/generated/ --real_dir backgrounds/ \\
        --output_dir synthetic_styled/ --config RIMES
"""

import sys
import math
import argparse
import re
import pickle
import shutil
import unicodedata
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from html import escape as xml_escape, unescape

import numpy as np
from PIL import Image, ImageFilter

SCRABBLEGAN_DIR = Path("./scrabblegan_arshjot")

GREEN  = "\033[32m"
RED    = "\033[31m"
ORANGE = "\033[33m"
RESET  = "\033[0m"

def _ok(msg):   return f"{GREEN}{msg}{RESET}"
def _err(msg):  return f"{RED}{msg}{RESET}"
def _warn(msg): return f"{ORANGE}{msg}{RESET}"


# =============================================================================
# UTILITAIRES COMMUNS
# =============================================================================

def find_pairs(data_dir: str) -> list:
    pairs = []
    for xml in sorted(Path(data_dir).glob("**/*.xml")):
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            img = xml.with_suffix(ext)
            if img.exists():
                pairs.append((str(xml), str(img)))
                break
    return pairs


def parse_alto(alto_path: str) -> list:
    tree = ET.parse(alto_path)
    root = tree.getroot()
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    lines = []
    for tl in root.iter(f"{ns}TextLine"):
        text = unescape(" ".join(
            s.get("CONTENT", "") for s in tl.iter(f"{ns}String")).strip())
        if text:
            lines.append({"id": tl.get("ID", ""), "text": text})
    return lines


def extract_texts(alto_dir: str) -> list:
    texts = set()
    for xml in sorted(Path(alto_dir).glob("**/*.xml")):
        for line in parse_alto(str(xml)):
            texts.add(line["text"])
    return sorted(texts)


# =============================================================================
# ÉTAPE 1 — CONTRAST ENHANCE
# =============================================================================

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def step_contrast(input_dir: str, output_dir: str, gamma: float):
    import cv2
    print("=== CONTRAST ENHANCE ===")
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_files = [f for f in sorted(input_dir.iterdir()) if f.suffix.lower() in IMG_EXTS]
    xml_files = sorted(input_dir.glob("*.xml"))
    print(f"Gamma : {gamma} | Images : {len(img_files)}, XML : {len(xml_files)}")

    ok_count, skipped = 0, 0
    for f in img_files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [!] Impossible de lire {f.name}"); skipped += 1; continue
        img = img.astype(np.float32) / 255.0
        img = np.power(img, gamma)
        cv2.imwrite(str(output_dir / f.name), (img * 255).clip(0, 255).astype(np.uint8))
        ok_count += 1

    for f in xml_files:
        shutil.copy(f, output_dir / f.name)

    print(f"-> {ok_count} images traitées, {len(xml_files)} XML copiés, {skipped} ignorées")
    print(f"-> {output_dir}")


# =============================================================================
# ÉTAPE 2 — WORDLEVEL (alignement Kraken)
# =============================================================================

def _safe_int(val, default=0) -> int:
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else int(f)
    except (TypeError, ValueError):
        return default


def _levenshtein_align(gt: str, ocr: str):
    m, n = len(gt), len(ocr)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] if gt[i-1] == ocr[j-1] \
                else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    align_gt, align_ocr = [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if gt[i-1] == ocr[j-1] else 1):
            align_gt.append(gt[i-1]); align_ocr.append(ocr[j-1]); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            align_gt.append(gt[i-1]); align_ocr.append(None); i -= 1
        else:
            align_gt.append(None); align_ocr.append(ocr[j-1]); j -= 1
    align_gt.reverse(); align_ocr.reverse()
    return align_gt, align_ocr


def _map_gt_to_cuts(gt: str, ocr: str, cuts: list) -> list:
    align_gt, align_ocr = _levenshtein_align(gt, ocr)
    ocr_idx = 0
    gt_cuts = []
    last_cut = cuts[0] if cuts else [[0,0],[0,0],[0,0],[0,0]]
    for ag, ao in zip(align_gt, align_ocr):
        if ag is None:
            if ocr_idx < len(cuts): last_cut = cuts[ocr_idx]
            ocr_idx += 1
        elif ao is None:
            gt_cuts.append(last_cut)
        else:
            if ocr_idx < len(cuts): last_cut = cuts[ocr_idx]
            gt_cuts.append(last_cut)
            ocr_idx += 1
    return gt_cuts


def _cuts_to_bbox(cuts: list) -> tuple:
    all_pts = [pt for cut in cuts for pt in cut]
    xs = [p[0] for p in all_pts if not (isinstance(p[0], float) and (math.isnan(p[0]) or math.isinf(p[0])))]
    ys = [p[1] for p in all_pts if not (isinstance(p[1], float) and (math.isnan(p[1]) or math.isinf(p[1])))]
    if not xs or not ys: return 0, 0, 1, 1
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _parse_points(s: str) -> list:
    pts = list(map(int, s.strip().split()))
    return [[pts[i], pts[i+1]] for i in range(0, len(pts)-1, 2)]


def _parse_alto_for_wordlevel(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns_raw = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
    ns = f"{{{ns_raw}}}" if ns_raw else ""
    page = root.find(f".//{ns}Page")
    page_w = _safe_int(page.get("WIDTH", 0))
    page_h = _safe_int(page.get("HEIGHT", 0))
    img_name = root.findtext(f".//{ns}fileName", "")
    lines = []
    skipped = 0
    for tl in root.iter(f"{ns}TextLine"):
        strings = list(tl.findall(f"{ns}String"))
        gt_text = " ".join(s.get("CONTENT", "").strip() for s in strings if s.get("CONTENT", "").strip())
        if not gt_text: continue
        baseline_str = tl.get("BASELINE", "")
        shape = tl.find(f"{ns}Shape")
        poly_el = shape.find(f"{ns}Polygon") if shape is not None else None
        poly_str = poly_el.get("POINTS", "") if poly_el is not None else ""
        if not baseline_str or not poly_str: skipped += 1; continue
        lines.append({
            'id': tl.get("ID", f"line_{len(lines)}"),
            'baseline': _parse_points(baseline_str),
            'boundary': _parse_points(poly_str),
            'gt_text':  gt_text,
            'tags':     tl.get("TAGREFS", ""),
            'hpos':     _safe_int(tl.get("HPOS", 0)),
            'vpos':     _safe_int(tl.get("VPOS", 0)),
            'width':    _safe_int(tl.get("WIDTH", 0)),
            'height':   _safe_int(tl.get("HEIGHT", 0)),
        })
    return lines, page_w, page_h, img_name, skipped


def _recognize_lines(img: Image.Image, lines: list, net) -> list:
    from kraken import rpred
    from kraken.containers import Segmentation, BaselineLine
    results = []
    for line in lines:
        def clean_cut(cut):
            return [[0, 0] if (math.isnan(float(p[0])) or math.isnan(float(p[1]))
                             or math.isinf(float(p[0])) or math.isinf(float(p[1])))
                    else [int(p[0]), int(p[1])] for p in cut]
        try:
            seg = Segmentation(
                type='baselines', imagename='',
                lines=[BaselineLine(id=line['id'], baseline=line['baseline'],
                                    boundary=line['boundary'], text=None)],
                regions={}, line_orders=[], text_direction='horizontal-lr',
                script_detection=False)
            for record in rpred.rpred(net, img, seg):
                results.append({**line, 'ocr_text': record.prediction,
                                 'cuts': [clean_cut(c) for c in record.cuts]})
        except Exception as e:
            gt_preview = line['gt_text'][:30]
            print(f"\n  {_err(f'[!] Echec ligne {gt_preview!r} : {e}')}")
            results.append({**line, 'ocr_text': None, 'cuts': []})
    return results


def _build_word_bboxes(gt_text: str, ocr_text: Optional[str], cuts: list) -> list:
    if not ocr_text or not cuts: return []
    gt_cuts = _map_gt_to_cuts(gt_text, ocr_text, cuts)
    words, current_word, current_cuts = [], [], []
    for char, cut in zip(gt_text, gt_cuts):
        if char == ' ':
            if current_word:
                words.append((''.join(current_word), current_cuts[:]))
                current_word, current_cuts = [], []
        else:
            current_word.append(char); current_cuts.append(cut)
    if current_word: words.append((''.join(current_word), current_cuts))
    result = []
    for word, word_cuts in words:
        if not word_cuts: continue
        x1, y1, x2, y2 = _cuts_to_bbox(word_cuts)
        result.append({'word': word, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                       'char_cuts': word_cuts})
    return result


def _build_alto_wordlevel_xml(lines_data: list, page_w: int, page_h: int, img_name: str) -> str:
    def sc(v):
        try:
            f = float(v); return 0 if (math.isnan(f) or math.isinf(f)) else int(f)
        except: return 0
    def pts(lst): return ' '.join(f"{sc(p[0])} {sc(p[1])}" for p in lst)
    def e(s): return xml_escape(str(s), quote=True)

    lines_xml = []
    for line in lines_data:
        word_bboxes = line.get('word_bboxes', [])
        if word_bboxes:
            strings_xml = []
            for w_idx, wb in enumerate(word_bboxes):
                w_w = max(1, wb['x2'] - wb['x1']); w_h = max(1, wb['y2'] - wb['y1'])
                glyphs = []
                for c_idx, (char, cut) in enumerate(zip(wb['word'], wb['char_cuts'])):
                    cx1, cy1, cx2, cy2 = _cuts_to_bbox([cut])
                    cw = max(1, cx2-cx1); ch = max(1, cy2-cy1)
                    glyphs.append(
                        f'              <Glyph ID="char_{w_idx}_{c_idx}" CONTENT="{e(char)}"\n'
                        f'                     HPOS="{cx1}" VPOS="{cy1}" WIDTH="{cw}" HEIGHT="{ch}" GC="1.0000">\n'
                        f'                <Shape><Polygon POINTS="{pts(cut)}"/></Shape>\n'
                        f'              </Glyph>')
                strings_xml.append(
                    f'            <String ID="segment_{w_idx}" CONTENT="{e(wb["word"])}"\n'
                    f'                    HPOS="{wb["x1"]}" VPOS="{wb["y1"]}" WIDTH="{w_w}" HEIGHT="{w_h}" WC="1.0000">\n'
                    + '\n'.join(glyphs) + '\n            </String>')
            content_xml = '\n'.join(strings_xml)
        else:
            content_xml = (
                f'            <String CONTENT="{e(line["gt_text"])}"\n'
                f'                    HPOS="{line["hpos"]}" VPOS="{line["vpos"]}"\n'
                f'                    WIDTH="{line["width"]}" HEIGHT="{line["height"]}"></String>')
        tagrefs = f'\n                    TAGREFS="{line["tags"]}"' if line['tags'] else ''
        lines_xml.append(
            f'          <TextLine ID="{line["id"]}"{tagrefs}\n'
            f'                    BASELINE="{pts(line["baseline"])}"\n'
            f'                    HPOS="{line["hpos"]}" VPOS="{line["vpos"]}"\n'
            f'                    WIDTH="{line["width"]}" HEIGHT="{line["height"]}">\n'
            f'            <Shape><Polygon POINTS="{pts(line["boundary"])}"/></Shape>\n'
            + content_xml + '\n          </TextLine>')

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<alto xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
        '      xmlns="http://www.loc.gov/standards/alto/ns-v4#"\n'
        '      xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v4#'
        ' http://www.loc.gov/standards/alto/v4/alto-4-2.xsd">\n'
        '  <Description><MeasurementUnit>pixel</MeasurementUnit>\n'
        f'    <sourceImageInformation><fileName>{e(img_name)}</fileName></sourceImageInformation>\n'
        '  </Description>\n  <Layout>\n'
        f'    <Page WIDTH="{page_w}" HEIGHT="{page_h}" PHYSICAL_IMG_NR="0" ID="eSc_dummypage_">\n'
        f'      <PrintSpace HPOS="0" VPOS="0" WIDTH="{page_w}" HEIGHT="{page_h}">\n'
        '        <TextBlock ID="eSc_dummyblock_">\n'
        + '\n'.join(lines_xml) + '\n'
        + '        </TextBlock>\n      </PrintSpace>\n    </Page>\n  </Layout>\n</alto>'
    )


_kraken_model_cache = {}


def step_wordlevel(xml_dir: str, model_path: str, output_dir: str = None,
                   img_dir: str = None, verbose: bool = False):
    from kraken.lib import models
    print("=== WORDLEVEL ===")
    xml_dir = Path(xml_dir)
    img_dir = Path(img_dir) if img_dir else xml_dir
    out_dir = Path(output_dir) if output_dir else None

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        import zipfile, datetime
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = xml_dir.parent / f"{xml_dir.name}_backup_{stamp}.zip"
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for p in xml_dir.glob("*.xml"): zf.write(p, p.name)
        print(f"Backup : {backup_path}")

    if model_path not in _kraken_model_cache:
        _kraken_model_cache[model_path] = models.load_any(model_path)
    net = _kraken_model_cache[model_path]

    xml_files = sorted(xml_dir.glob('*.xml'))
    print(f"Traitement de {len(xml_files)} fichiers...")
    for xml_path in xml_files:
        img_path = None
        for ext in IMG_EXTS:
            p = img_dir / xml_path.with_suffix(ext).name
            if p.exists(): img_path = p; break
        if img_path is None:
            print(f"  {_err(f'[!] Image non trouvée pour {xml_path.name}')}"); continue
        print(f"  {xml_path.name}", end='', flush=True)
        try:
            lines, page_w, page_h, img_name, skipped = _parse_alto_for_wordlevel(str(xml_path))
        except ET.ParseError as e:
            print(f"\n  {_err(f'[!] XML invalide : {e}')}"); continue
        if skipped: print(f" {_warn(f'({skipped} lignes sans baseline)')}", end='')
        if not lines: print(f"\n  {_warn('[!] Aucune ligne')}"); continue

        img = Image.open(img_path)
        recognized = _recognize_lines(img, lines, net)
        n_failed = 0
        for rec in recognized:
            if verbose:
                print(f"\n    GT  : '{rec['gt_text']}'")
                print(f"    OCR : '{rec['ocr_text']}'")
            rec['word_bboxes'] = _build_word_bboxes(rec['gt_text'], rec['ocr_text'], rec['cuts'])
            if not rec['word_bboxes']: n_failed += 1

        alto_xml = _build_alto_wordlevel_xml(recognized, page_w, page_h, img_name)
        out_path = (out_dir / xml_path.name) if out_dir else xml_path
        out_path.write_text(alto_xml, encoding='utf-8')
        n_words = sum(len(r['word_bboxes']) for r in recognized)
        if n_failed == 0:
            status = _ok(f"-> {len(recognized)} lignes, {n_words} mots")
        elif n_failed < len(recognized):
            status = _warn(f"-> {len(recognized)} lignes, {n_words} mots ({n_failed} sans positions)")
        else:
            status = _err("-> aucune position générée")
        print(f"\n  {status}")


# =============================================================================
# ÉTAPE 3 — NORMALIZE
# =============================================================================

def _load_charmap(charmap_path: str) -> set:
    with open(charmap_path, 'rb') as f:
        cm = pickle.load(f, encoding='latin1')
    chars = set(cm.keys()) if isinstance(cm, dict) else set(cm)
    chars = {str(c) for c in chars if len(str(c)) == 1}
    chars.add(' ')
    return chars


def _strip_diacritics(text: str) -> str:
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')


def _filter_text(text: str, valid_chars: set, report_changes: list = None) -> str:
    decoded = unicodedata.normalize('NFC', unescape(text))
    result = []
    for c in decoded:
        if c in valid_chars:
            result.append(c)
        else:
            fallback = _strip_diacritics(c)
            if fallback and fallback in valid_chars:
                result.append(fallback)
    filtered = re.sub(r' +', ' ', ''.join(result)).strip()
    filtered_encoded = xml_escape(filtered, quote=True)
    if report_changes is not None and text != filtered_encoded:
        report_changes.append((text, filtered_encoded))
    return filtered_encoded


def step_normalize(xml_dir: str, charmap: str, output_dir: str = None, report: bool = False):
    print("=== NORMALIZE ===")
    valid_chars = _load_charmap(charmap)
    print(f"Char_map : {len(valid_chars)} caractères valides")
    xml_dir    = Path(xml_dir)
    output_dir = Path(output_dir) if output_dir else None
    if output_dir: output_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(xml_dir.glob('*.xml'))
    print(f"Traitement de {len(xml_files)} fichiers...")
    for xml_path in xml_files:
        out_path = (output_dir / xml_path.name) if output_dir else xml_path
        try:
            content = xml_path.read_text(encoding='utf-8')
            changes = []
            def replace_content(m):
                return 'CONTENT="' + _filter_text(m.group(1), valid_chars,
                    report_changes=changes if report else None) + '"'
            result = re.sub(r'CONTENT="([^"]*)"', replace_content, content)
            out_path.write_text(result, encoding='utf-8')
            if report and changes:
                print(f"  {xml_path.name}")
                for before, after in changes:
                    print(f"    {before!r} -> {after!r}")
        except Exception as ex:
            print(f"  [!] {xml_path.name} : {ex}")
    print(f"-> {len(xml_files)} fichiers traités")


# =============================================================================
# ÉTAPE 4 — FINETUNE
# =============================================================================

def step_finetune(weights: str, charmap: str, alto_dir: str, output_dir: str,
                  epochs: int, save_patches_dir=None):
    print("=== FINE-TUNING ===")
    output_dir = Path(output_dir)
    ft_dir     = output_dir / "finetuned"
    ft_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Chargement du character mapping : {charmap}")
    with open(charmap, "rb") as f:
        char_map = pickle.load(f, encoding="latin1")
    char2idx = char_map if isinstance(char_map, dict) else {c: i for i, c in enumerate(char_map)}
    print(f"  -> {len(char2idx)} caractères")

    print("[2/4] Extraction des patches ALTO...")
    if save_patches_dir:
        save_patches_dir = Path(save_patches_dir)
        save_patches_dir.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(alto_dir)
    if not pairs:
        print(f"[!] Aucune paire ALTO/image dans {alto_dir}"); return

    samples = []
    for xml_path, img_path in pairs:
        try:
            img_full = Image.open(img_path).convert("L")
            w_img, h_img = img_full.size
            tree = ET.parse(xml_path)
            root = tree.getroot()
            ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
        except ET.ParseError:
            print(f"\n❌ XML invalide: {xml_path}"); continue

        for s in root.iter(f"{ns}String"):
            text = unescape(s.get("CONTENT", "").strip())
            if not text: continue
            text = "".join(c for c in text if c in char2idx).strip()
            if not text: continue
            hpos   = int(float(s.get("HPOS",   0)))
            vpos   = int(float(s.get("VPOS",   0)))
            width  = int(float(s.get("WIDTH",  0)))
            height = int(float(s.get("HEIGHT", 0)))
            if width <= 0 or height <= 0: continue
            x1, y1 = max(0, hpos), max(0, vpos)
            x2, y2 = min(w_img, hpos + width), min(h_img, vpos + height)
            if x2 <= x1 or y2 <= y1: continue
            patch = img_full.crop((x1, y1, x2, y2))
            new_w = max(1, int(patch.width * 32 / patch.height))
            patch = patch.resize((new_w, 32), Image.LANCZOS)
            samples.append((np.array(patch), text))
            if save_patches_dir:
                safe    = text[:30].replace(" ", "_").replace("/", "-")
                idx_str = f"{len(samples):05d}"
                patch_path = save_patches_dir / f"{idx_str}_{safe}.png"
                patch.save(patch_path)
                patch_path.with_suffix(".txt").write_text(text, encoding="utf-8")

    print(f"  -> {len(samples)} patches extraits")

    print("[3/4] Préparation du dataset pickle...")
    data_pkl  = ft_dir / "custom_data.pkl"
    word_data = {}; skipped = 0
    for i, (arr, text) in enumerate(samples):
        label = [char2idx[c] for c in text if c in char2idx]
        if not label: skipped += 1; continue
        word_data[i] = (label, arr)
    with open(data_pkl, "wb") as f:
        pickle.dump({"word_data": word_data, "char_map": char2idx}, f)
    print(f"  -> {len(word_data)} échantillons ({skipped} ignorés) -> {data_pkl}")

    ckpt_copy = ft_dir / "checkpoint_finetune.pt"
    if not ckpt_copy.exists():
        shutil.copy(weights, ckpt_copy)

    config_path   = SCRABBLEGAN_DIR / "config.py"
    config_backup = SCRABBLEGAN_DIR / "config_backup.py"
    if config_backup.exists():
        original_config = config_backup.read_text(encoding="utf-8")
    else:
        original_config = config_path.read_text(encoding="utf-8")
        config_backup.write_text(original_config, encoding="utf-8")

    data_pkl_abs = str(data_pkl.resolve())
    ckpt_dir_abs = str(ckpt_copy.parent.resolve())
    lexicon_abs  = str((SCRABBLEGAN_DIR / 'data' / 'Lexicon' / 'Lexique383.tsv').resolve())

    config_path.write_text(f"""
# === PATCH FINETUNE (ajouté automatiquement) ===
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
    device = _torch.device('cuda' if _torch.cuda.is_available() else 'cpu')
""", encoding="utf-8")
    print(f"  -> config.py réécrit")

    weights_dir = SCRABBLEGAN_DIR.resolve() / "weights"
    weights_dir.mkdir(exist_ok=True)
    for old_ckpt in weights_dir.glob("*.pth.tar"):
        old_ckpt.unlink()
        print(f"  -> Ancien checkpoint supprimé : {old_ckpt.name}")
    ckpt_in_weightdir = weights_dir / "model_checkpoint_epoch_0.pth.tar"
    shutil.copy(weights, ckpt_in_weightdir)
    print(f"  -> Checkpoint copié -> {ckpt_in_weightdir}")

    print(f"[4/4] Lancement de train.py ({epochs} epochs)...")
    result = subprocess.run(
        [sys.executable, str(SCRABBLEGAN_DIR.resolve() / "train.py")],
        cwd=str(SCRABBLEGAN_DIR.resolve()))
    config_path.write_text(original_config, encoding="utf-8")
    print(f"  -> config.py restauré")

    if result.returncode == 0:
        print(f"\n[OK] Fine-tuning terminé -> {ft_dir}")
    else:
        print(f"\n[!] train.py a retourné une erreur (voir ci-dessus)")


# =============================================================================
# ÉTAPE 5 — GENERATE
# =============================================================================

def step_generate(weights: str, charmap: str, alto_dir: str, output_dir: str, n_images: int):
    print("=== GENERATION ===")

    if str(SCRABBLEGAN_DIR.resolve()) not in sys.path:
        sys.path.insert(0, str(SCRABBLEGAN_DIR.resolve()))

    print("[1/3] Extraction des textes depuis les fichiers ALTO...")
    texts = extract_texts(alto_dir)
    if not texts:
        print(f"[!] Aucun texte trouvé dans {alto_dir}"); return

    output_dir = Path(output_dir)
    gen_dir    = output_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    print("[2/3] Chargement du modèle...")
    with open(charmap, "rb") as _f:
        char_map = pickle.load(_f, encoding="latin1")
    valid_chars   = set(char_map.keys()) if isinstance(char_map, dict) else set(char_map)
    num_chars_val = len(char_map) if isinstance(char_map, dict) else len(valid_chars)

    def clean(t): return "".join(c for c in t if c in valid_chars).strip()
    texts_clean = [(orig, clean(orig)) for orig in texts if clean(orig)]
    print(f"  -> {len(texts_clean)} textes valides sur {len(texts)}")

    config_path   = SCRABBLEGAN_DIR / "config.py"
    config_txt    = config_path.read_text(encoding="utf-8")
    lex_abs       = str((SCRABBLEGAN_DIR / "data" / "Lexicon" / "Lexique383.tsv").resolve())
    config_patched = re.sub(r"lexicon_file\s*=.*", f"lexicon_file = r'{lex_abs}'", config_txt)
    config_patched = re.sub(r"num_chars\s*=.*", f"num_chars = {num_chars_val}", config_patched)
    config_path.write_text(config_patched, encoding="utf-8")

    try:
        for mod in list(sys.modules.keys()):
            if mod in ("config", "generate_images"): del sys.modules[mod]
        from generate_images import ImgGenerator
        from config import Config
        Config.num_chars = num_chars_val
        generator = ImgGenerator(checkpt_path=str(Path(weights).resolve()),
                                 config=Config, char_map=char_map)
    except Exception as e:
        print(f"[!] Impossible de charger ImgGenerator : {e}")
        config_path.write_text(config_txt, encoding="utf-8"); return

    _pairs = find_pairs(alto_dir)
    _samples = []
    for _, img_path in _pairs[:10]:
        try:
            arr_s = np.array(Image.open(img_path).convert("RGB"))
            _samples.append(tuple(int(np.percentile(arr_s[:,:,c], 95)) for c in range(3)))
        except Exception: pass
    bg_color = tuple(int(np.mean([s[c] for s in _samples])) for c in range(3)) \
               if _samples else (220, 210, 185)
    print(f"  bg_color calculé sur {len(_samples)} images : {bg_color}")

    def normalize_img(img_arr, bg=bg_color):
        arr = np.array(img_arr, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = arr ** 1.8
        return np.stack([(arr * bg[i]).astype(np.uint8) for i in range(3)], axis=2)

    def blend_join(img_a, img_b, space=16):
        h = img_a.shape[0]
        col_a = img_a[:, -4:, :].astype(np.float32).mean(axis=(0,1))
        col_b = img_b[:, :4, :].astype(np.float32).mean(axis=(0,1))
        mid   = ((col_a + col_b) / 2).astype(np.uint8)
        gap   = np.stack([np.full((h, space), mid[i], dtype=np.uint8) for i in range(3)], axis=2)
        fade  = min(6, img_a.shape[1] // 2, space // 2)
        a = img_a.astype(np.float32); g = gap.astype(np.float32)
        w = np.linspace(1, 0, fade)[np.newaxis, :, np.newaxis]
        mask_a = (a[:, -fade:].mean(axis=2) > 120)[:, :, np.newaxis]
        g[:, :fade] = np.where(mask_a, a[:, -fade:] * w + g[:, :fade] * (1-w), g[:, :fade])
        b  = img_b.astype(np.float32)
        w2 = np.linspace(0, 1, fade)[np.newaxis, :, np.newaxis]
        mask_b = (b[:, :fade].mean(axis=2) > 120)[:, :, np.newaxis]
        g[:, -fade:] = np.where(mask_b, g[:, -fade:] * (1-w2) + b[:, :fade] * w2, g[:, -fade:])
        return np.concatenate([img_a, g.astype(np.uint8), img_b], axis=1)

    def generate_line_image(generator, text, char_map, n_variants):
        words_clean = [(w, "".join(c for c in w if c in char_map))
                       for w in text.split() if "".join(c for c in w if c in char_map)]
        if not words_clean: return []
        results = []
        for _ in range(n_variants):
            word_imgs = []
            for orig_w, clean_w in words_clean:
                try:
                    imgs, _, _ = generator.generate(random_num_imgs=1, word_list=[clean_w])
                    word_imgs.append((orig_w, normalize_img(imgs[0])))
                except Exception as e:
                    print(f"  Error: {e}")
            if not word_imgs: continue
            word_positions = []; x_cursor = 0; line_img = None
            for i, (orig_w, wimg) in enumerate(word_imgs):
                w_width = wimg.shape[1]
                if i == 0:
                    line_img = wimg
                    word_positions.append((orig_w, 0, w_width))
                    x_cursor = w_width
                else:
                    x1 = x_cursor + 16
                    line_img = blend_join(line_img, wimg)
                    word_positions.append((orig_w, x1, x1 + w_width))
                    x_cursor = x1 + w_width
            results.append((line_img, word_positions))
        return results

    def build_alto_synth(img_path: Path, orig_text: str, word_positions: list,
                         img_w: int, img_h: int, line_idx: int) -> str:
        line_id   = f"eSc_line_{line_idx:06d}"
        block_id  = f"eSc_block_{line_idx:06d}"
        baseline  = f"0 {int(img_h * 0.80)} {img_w} {int(img_h * 0.80)}"
        poly_line = f"0 0 0 {img_h} {img_w} {img_h} {img_w} 0"
        strings_xml = []
        for w_idx, (word, x1, x2) in enumerate(word_positions):
            w_width = max(1, x2 - x1)
            glyphs = []
            for c_idx, char in enumerate(word):
                char_x = x1 + int(w_width * c_idx / max(1, len(word)))
                char_w = max(1, w_width // max(1, len(word)))
                gp = f"{char_x} 0 {char_x} {img_h} {char_x+char_w} {img_h} {char_x+char_w} 0"
                glyphs.append(
                    f'              <Glyph ID="char_{w_idx}_{c_idx}" CONTENT="{char}"\n'
                    f'                     HPOS="{char_x}" VPOS="0" WIDTH="{char_w}" HEIGHT="{img_h}" GC="1.0000">\n'
                    f'                <Shape><Polygon POINTS="{gp}"/></Shape>\n'
                    f'              </Glyph>')
            strings_xml.append(
                f'            <String ID="segment_{w_idx}" CONTENT="{word}"\n'
                f'                    HPOS="{x1}" VPOS="0" WIDTH="{w_width}" HEIGHT="{img_h}" WC="1.0000">\n'
                + ''.join(glyphs) + '\n            </String>')
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<alto xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
            '      xmlns="http://www.loc.gov/standards/alto/ns-v4#"\n'
            '      xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v4#'
            ' http://www.loc.gov/standards/alto/v4/alto-4-2.xsd">\n'
            '  <Description><MeasurementUnit>pixel</MeasurementUnit>\n'
            f'    <sourceImageInformation><fileName>{img_path.name}</fileName></sourceImageInformation>\n'
            '  </Description>\n'
            f'  <Layout><Page WIDTH="{img_w}" HEIGHT="{img_h}" PHYSICAL_IMG_NR="{line_idx}" ID="eSc_dummypage_">\n'
            f'    <PrintSpace HPOS="0" VPOS="0" WIDTH="{img_w}" HEIGHT="{img_h}">\n'
            f'      <TextBlock ID="{block_id}">\n'
            f'        <TextLine ID="{line_id}" TAGREFS="LT1047" BASELINE="{baseline}"\n'
            f'                  HPOS="0" VPOS="0" WIDTH="{img_w}" HEIGHT="{img_h}">\n'
            f'          <Shape><Polygon POINTS="{poly_line}"/></Shape>\n'
            + ''.join(strings_xml) + '\n'
            + '        </TextLine></TextBlock></PrintSpace></Page></Layout></alto>'
        )

    print(f"[3/3] Génération de {len(texts_clean) * n_images} images...")
    manifest = []; failed = 0
    for idx, (orig_text, clean_text) in enumerate(texts_clean):
        try:
            results = generate_line_image(generator, orig_text, char_map, n_images)
            safe = orig_text[:50].replace(" ", "_").replace("/", "-")
            for i, (arr, word_positions) in enumerate(results):
                img_path = gen_dir / f"{idx:05d}_{safe}_{i:03d}.png"
                pil_img  = Image.fromarray(arr)
                pil_img.save(img_path)
                img_w, img_h = pil_img.size
                img_path.with_suffix(".txt").write_text(orig_text, encoding="utf-8")
                img_path.with_suffix(".xml").write_text(
                    build_alto_synth(img_path, orig_text, word_positions, img_w, img_h,
                                     idx * n_images + i), encoding="utf-8")
                manifest.append((str(img_path), orig_text))
        except Exception as e:
            failed += 1
            if failed <= 3: print(f"  [!] Echec '{orig_text[:30]}' : {e}")
        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(texts_clean)} traités ({len(manifest)} images)...")

    manifest_path = output_dir / "manifest_generated.csv"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("image_path\ttext\n")
        for p, t in manifest: f.write(f"{p}\t{t}\n")
    config_path.write_text(config_txt, encoding="utf-8")
    print(f"\n[OK] {len(manifest)} images -> {gen_dir}")
    if failed: print(f"  ({failed} textes ont échoué)")
    print(f"  Manifeste : {manifest_path}")


# =============================================================================
# ÉTAPE 6 — STYLE TRANSFER
# =============================================================================

STYLE_CONFIGS = {
    'IAM': {
        'bg_percentile':  80,
        'ink_percentile': 15,
        'ink_darken':     0.5,
        'mask_blur':      0.6,
        'mask_gamma':     1.5,
    },
    'RIMES': {
        'bg_percentile':  85,
        'ink_percentile': 10,
        'ink_darken':     0.15,
        'mask_blur':      0.4,
        'mask_gamma':     2.5,
    },
}


def _sample_colors(bg: Image.Image, cfg: dict) -> tuple:
    arr  = np.array(bg.convert('RGB')).astype(float)
    gray = arr.mean(axis=2)
    bg_mask  = gray > np.percentile(gray, cfg['bg_percentile'])
    ink_mask = gray < np.percentile(gray, cfg['ink_percentile'])
    bg_color  = arr[bg_mask].mean(axis=0)  if bg_mask.any()  else np.array([220., 210., 185.])
    ink_color = arr[ink_mask].mean(axis=0) * cfg['ink_darken'] \
                if ink_mask.any() else np.array([30., 25., 20.])
    return bg_color, ink_color


def _compose_style(synth_path: str, backgrounds: list, cfg: dict) -> Image.Image:
    import random
    synth     = Image.open(synth_path).convert('L')
    synth_arr = np.array(synth).astype(float)
    s_min, s_max = synth_arr.min(), synth_arr.max()
    synth_norm = (synth_arr - s_min) / (s_max - s_min) * 255 if s_max > s_min else synth_arr.copy()
    mask_pil = Image.fromarray(synth_norm.astype(np.uint8))
    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=cfg['mask_blur']))
    mask  = np.array(mask_pil).astype(float) / 255.0
    mask  = mask ** cfg.get('mask_gamma', 1.5)
    mask3 = np.stack([mask, mask, mask], axis=2)
    bg_color, ink_color = _sample_colors(random.choice(backgrounds), cfg)
    result = mask3 * bg_color + (1 - mask3) * ink_color
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode='RGB')


def step_style(synth_dir: str, real_dir: str, output_dir: str, config: str = 'IAM'):
    print("=== STYLE TRANSFER ===")
    cfg = STYLE_CONFIGS[config]
    print(f"Config : {config} (ink_darken={cfg['ink_darken']}, mask_gamma={cfg['mask_gamma']})")

    real_dir = Path(real_dir)
    backgrounds = []
    for ext in IMG_EXTS:
        for p in real_dir.glob(f'*{ext}'):
            try: backgrounds.append(Image.open(p).convert('RGB'))
            except Exception as e: print(f"  [!] {p.name} : {e}")
    if not backgrounds:
        raise ValueError(f"Aucune image trouvée dans {real_dir}")
    print(f"  -> {len(backgrounds)} fonds chargés")

    synth_dir = Path(synth_dir)
    out_dir   = Path(output_dir) if output_dir else synth_dir.parent / (synth_dir.name + '_styled')
    out_dir.mkdir(parents=True, exist_ok=True)

    synth_files = sorted(synth_dir.glob('*.png'))
    print(f"Traitement de {len(synth_files)} images...")
    for synth_path in synth_files:
        try:
            img = _compose_style(str(synth_path), backgrounds, cfg)
            img.save(out_dir / synth_path.name)
        except Exception as e:
            print(f"  [!] {synth_path.name} : {e}")
    for txt_path in synth_dir.glob('*.txt'):
        (out_dir / txt_path.name).write_text(
            txt_path.read_text(encoding='utf-8'), encoding='utf-8')
    print(f"-> {out_dir} ({len(synth_files)} images)")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ScrabbleGAN — pipeline HTR français historique",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--step", required=True,
                        choices=["contrast", "wordlevel", "normalize", "finetune", "generate", "style"],
                        help="Étape à exécuter")

    # contrast
    parser.add_argument("--input_dir",  help="[contrast] Dossier images en entrée")
    parser.add_argument("--gamma",      type=float, default=0.8,
                        help="[contrast] Valeur gamma (<1=assombrit, >1=éclaircit)")

    # wordlevel
    parser.add_argument("--xml_dir",    help="[wordlevel/normalize] Dossier ALTO en entrée")
    parser.add_argument("--img_dir",    help="[wordlevel] Dossier images (défaut = xml_dir)")
    parser.add_argument("--model",      help="[wordlevel] Modèle Kraken (.mlmodel)")
    parser.add_argument("--verbose",    action="store_true",
                        help="[wordlevel] Détails GT/OCR ligne par ligne")

    # normalize
    parser.add_argument("--charmap",    help="[normalize/finetune/generate] Char_map pickle")
    parser.add_argument("--report",     action="store_true",
                        help="[normalize] Afficher les changements effectués")

    # finetune / generate
    parser.add_argument("--weights",    help="[finetune/generate] Checkpoint ScrabbleGAN")
    parser.add_argument("--alto_dir",   default="./data",
                        help="[finetune/generate] Dossier ALTO/image")
    parser.add_argument("--epochs",     type=int, default=10,
                        help="[finetune] Nombre d'epochs")
    parser.add_argument("--save_patches", default=None, metavar="DIR",
                        help="[finetune] Sauvegarder les patches dans ce dossier")
    parser.add_argument("--n_images",   type=int, default=5,
                        help="[generate] Images synthétiques par texte")

    # style
    parser.add_argument("--synth_dir",  help="[style] Dossier images synthétiques")
    parser.add_argument("--real_dir",   help="[style] Dossier fonds réels")
    parser.add_argument("--config",     default="IAM", choices=list(STYLE_CONFIGS.keys()),
                        help="[style] Config couleur : IAM ou RIMES")

    # commun
    parser.add_argument("--output_dir", default="./output",
                        help="Dossier de sortie")

    args = parser.parse_args()

    if args.step == "contrast":
        step_contrast(args.input_dir, args.output_dir, args.gamma)
    elif args.step == "wordlevel":
        step_wordlevel(args.xml_dir, args.model, args.output_dir,
                       img_dir=args.img_dir, verbose=args.verbose)
    elif args.step == "normalize":
        step_normalize(args.xml_dir, args.charmap, args.output_dir, report=args.report)
    elif args.step == "finetune":
        step_finetune(args.weights, args.charmap, args.alto_dir,
                      args.output_dir, args.epochs, save_patches_dir=args.save_patches)
    elif args.step == "generate":
        step_generate(args.weights, args.charmap, args.alto_dir,
                      args.output_dir, args.n_images)
    elif args.step == "style":
        step_style(args.synth_dir, args.real_dir, args.output_dir, config=args.config)


if __name__ == "__main__":
    main()
