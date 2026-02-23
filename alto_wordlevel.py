"""
alto_wordlevel.py - Alignement positions Kraken + texte GT pour ALTO word-level
================================================================================

Prend un ALTO eScriptorium (texte GT, pas de positions mot) et produit un
ALTO word-level avec :
  - les positions caractère/mot récupérées depuis Kraken
  - le texte GT conservé (pas le texte OCR)

Corrections :
  - Couleurs terminal : vert (OK), rouge (erreur), orange (aucune ligne)
  - Les lignes déjà word-level sont retraitées (pas skippées) pour intégrer
    les corrections manuelles de l'annotateur
  - Gestion des NaN dans HPOS/VPOS/WIDTH/HEIGHT
  - Les attributs numériques sont toujours des entiers (pas de floats)
  - Les caractères spéciaux XML dans CONTENT sont échappés (&quot; etc.)
  - Les lignes sans baseline/polygon sont skippées proprement

Usage :
    # Fichier unique (écrase l'original)
    python alto_wordlevel.py --xml data/page.xml --img data/page.jpg \\
        --model models/fondue_archimed_v4.mlmodel

    # Fichier unique avec sortie séparée
    python alto_wordlevel.py --xml data/page.xml --img data/page.jpg \\
        --model models/fondue_archimed_v4.mlmodel --output data/page_out.xml

    # Dossier entier (écrase les originaux par défaut)
    python alto_wordlevel.py --xml_dir data/ \\
        --model models/fondue_archimed_v4.mlmodel

    # Dossier entier avec sortie séparée
    python alto_wordlevel.py --xml_dir data/ --output_dir data_wordlevel/ \\
        --model models/fondue_archimed_v4.mlmodel
"""

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from html import escape as xml_escape

from PIL import Image
from kraken import rpred
from kraken.lib import models
from kraken.containers import Segmentation, BaselineLine


# =============================================================================
# COULEURS TERMINAL
# =============================================================================

GREEN  = "\033[32m"
RED    = "\033[31m"
ORANGE = "\033[33m"
RESET  = "\033[0m"

def ok(msg):    return f"{GREEN}{msg}{RESET}"
def err(msg):   return f"{RED}{msg}{RESET}"
def warn(msg):  return f"{ORANGE}{msg}{RESET}"


# =============================================================================
# 1. ALIGNEMENT LEVENSHTEIN
# =============================================================================

def levenshtein_align(gt: str, ocr: str):
    """
    Aligne deux chaînes via programmation dynamique (Levenshtein).
    Retourne deux listes de même longueur :
      - align_gt  : caractères GT  (None = insertion OCR sans correspondance GT)
      - align_ocr : caractères OCR (None = deletion, caractère GT sans correspondance)
    """
    m, n = len(gt), len(ocr)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt[i-1] == ocr[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    align_gt, align_ocr = [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if gt[i-1] == ocr[j-1] else 1):
            align_gt.append(gt[i-1])
            align_ocr.append(ocr[j-1])
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            align_gt.append(gt[i-1])
            align_ocr.append(None)
            i -= 1
        else:
            align_gt.append(None)
            align_ocr.append(ocr[j-1])
            j -= 1

    align_gt.reverse()
    align_ocr.reverse()
    return align_gt, align_ocr


def map_gt_to_cuts(gt: str, ocr: str, cuts: list) -> list:
    """
    Pour chaque caractère GT, retourne le cut Kraken correspondant.
    Deletions OCR → duplique le dernier cut connu.
    """
    align_gt, align_ocr = levenshtein_align(gt, ocr)
    ocr_idx = 0
    gt_cuts = []
    last_cut = cuts[0] if cuts else [[0, 0], [0, 0], [0, 0], [0, 0]]

    for ag, ao in zip(align_gt, align_ocr):
        if ag is None:
            if ocr_idx < len(cuts):
                last_cut = cuts[ocr_idx]
            ocr_idx += 1
        elif ao is None:
            gt_cuts.append(last_cut)
        else:
            if ocr_idx < len(cuts):
                last_cut = cuts[ocr_idx]
            gt_cuts.append(last_cut)
            ocr_idx += 1

    return gt_cuts


def cuts_to_bbox(cuts: list) -> tuple:
    """Convertit une liste de cuts en bbox (x1,y1,x2,y2) entiers.
    Filtre les coordonnées NaN/Inf produites par Kraken sur les baselines dégénérées.
    """
    import math
    all_pts = [pt for cut in cuts for pt in cut]
    xs = [p[0] for p in all_pts if not (isinstance(p[0], float) and (math.isnan(p[0]) or math.isinf(p[0])))]
    ys = [p[1] for p in all_pts if not (isinstance(p[1], float) and (math.isnan(p[1]) or math.isinf(p[1])))]
    if not xs or not ys:
        return 0, 0, 1, 1
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


# =============================================================================
# 2. PARSING ALTO
# =============================================================================

def parse_points(s: str) -> list:
    """Parse 'x0 y0 x1 y1 ...' en liste [[x,y], ...]."""
    pts = list(map(int, s.strip().split()))
    return [[pts[i], pts[i+1]] for i in range(0, len(pts)-1, 2)]


def safe_int(val, default=0) -> int:
    """Convertit en int en gérant NaN et None."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return int(f)
    except (TypeError, ValueError):
        return default


def get_gt_text(tl, ns: str) -> str:
    """
    Récupère le texte GT d'une TextLine, qu'elle soit line-level ou word-level.
    Pour word-level : concatène tous les <String> avec un espace.
    """
    strings = list(tl.findall(f"{ns}String"))
    if not strings:
        return ""
    return " ".join(s.get("CONTENT", "").strip() for s in strings if s.get("CONTENT", "").strip())


def parse_alto(xml_path: str):
    """
    Lit un ALTO eScriptorium.
    Traite toutes les lignes (y compris celles déjà word-level).
    Skippe seulement les lignes sans baseline ou polygon.
    Retourne (lines, page_w, page_h, img_name)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns_raw = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
    ns = f"{{{ns_raw}}}" if ns_raw else ""

    page = root.find(f".//{ns}Page")
    page_w = safe_int(page.get("WIDTH", 0))
    page_h = safe_int(page.get("HEIGHT", 0))
    img_name = root.findtext(f".//{ns}fileName", "")

    lines = []
    skipped_no_baseline = 0

    for tl in root.iter(f"{ns}TextLine"):
        gt_text = get_gt_text(tl, ns)
        if not gt_text:
            continue

        baseline_str = tl.get("BASELINE", "")
        shape = tl.find(f"{ns}Shape")
        poly_el = shape.find(f"{ns}Polygon") if shape is not None else None
        poly_str = poly_el.get("POINTS", "") if poly_el is not None else ""

        if not baseline_str or not poly_str:
            skipped_no_baseline += 1
            continue

        lines.append({
            'id':       tl.get("ID", f"line_{len(lines)}"),
            'baseline': parse_points(baseline_str),
            'boundary': parse_points(poly_str),
            'gt_text':  gt_text,
            'tags':     tl.get("TAGREFS", ""),
            'hpos':     safe_int(tl.get("HPOS",   0)),
            'vpos':     safe_int(tl.get("VPOS",   0)),
            'width':    safe_int(tl.get("WIDTH",  0)),
            'height':   safe_int(tl.get("HEIGHT", 0)),
        })

    return lines, page_w, page_h, img_name, skipped_no_baseline


# =============================================================================
# 3. RECONNAISSANCE KRAKEN
# =============================================================================

def recognize_lines(img: Image.Image, lines: list, net) -> list:
    """Fait tourner Kraken (baseline) sur chaque ligne."""
    results = []
    for line in lines:
        try:
            seg = Segmentation(
                type='baselines',
                imagename='',
                lines=[BaselineLine(
                    id=line['id'],
                    baseline=line['baseline'],
                    boundary=line['boundary'],
                    text=None
                )],
                regions={},
                line_orders=[],
                text_direction='horizontal-lr',
                script_detection=False
            )
            for record in rpred.rpred(net, img, seg):
                results.append({**line,
                    'ocr_text': record.prediction,
                    'cuts': record.cuts,
                })
        except Exception as e:
            gt_preview = line['gt_text'][:30]
            print(f"\n  {err(f'[!] Echec ligne {gt_preview!r} : {e}')}")
            results.append({**line, 'ocr_text': None, 'cuts': []})
    return results


# =============================================================================
# 4. CONSTRUCTION DES BBOXES MOT
# =============================================================================

def build_word_bboxes(gt_text: str, ocr_text: Optional[str], cuts: list) -> list:
    """Aligne GT et OCR, construit les bboxes de chaque mot GT."""
    if not ocr_text or not cuts:
        return []

    gt_cuts = map_gt_to_cuts(gt_text, ocr_text, cuts)

    words, current_word, current_cuts = [], [], []
    for char, cut in zip(gt_text, gt_cuts):
        if char == ' ':
            if current_word:
                words.append((''.join(current_word), current_cuts[:]))
                current_word, current_cuts = [], []
        else:
            current_word.append(char)
            current_cuts.append(cut)
    if current_word:
        words.append((''.join(current_word), current_cuts))

    result = []
    for word, word_cuts in words:
        if not word_cuts:
            continue
        x1, y1, x2, y2 = cuts_to_bbox(word_cuts)
        result.append({'word': word,
                       'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                       'char_cuts': word_cuts})
    return result


# =============================================================================
# 5. GENERATION ALTO WORD-LEVEL
# =============================================================================

def build_alto_xml(lines_data: list, page_w: int, page_h: int,
                   img_name: str) -> str:
    """Génère un ALTO v4 word-level — attributs numériques toujours entiers."""

    def pts_to_str(pts):
        return ' '.join(f"{int(p[0])} {int(p[1])}" for p in pts)

    def cut_to_poly(cut):
        return ' '.join(f"{int(p[0])} {int(p[1])}" for p in cut)

    def e(s):
        return xml_escape(str(s), quote=True)

    lines_xml = []
    for line in lines_data:
        gt_text     = line['gt_text']
        word_bboxes = line.get('word_bboxes', [])
        bl_str      = pts_to_str(line['baseline'])
        poly_str    = pts_to_str(line['boundary'])

        if word_bboxes:
            strings_xml = []
            for w_idx, wb in enumerate(word_bboxes):
                w_w = max(1, wb['x2'] - wb['x1'])
                w_h = max(1, wb['y2'] - wb['y1'])

                glyphs_xml = []
                for c_idx, (char, cut) in enumerate(zip(wb['word'], wb['char_cuts'])):
                    cx1, cy1, cx2, cy2 = cuts_to_bbox([cut])
                    cw = max(1, cx2 - cx1)
                    ch = max(1, cy2 - cy1)
                    glyphs_xml.append(
                        f'              <Glyph ID="char_{w_idx}_{c_idx}"\n'
                        f'                     CONTENT="{e(char)}"\n'
                        f'                     HPOS="{cx1}" VPOS="{cy1}"\n'
                        f'                     WIDTH="{cw}" HEIGHT="{ch}"\n'
                        f'                     GC="1.0000">\n'
                        f'                <Shape><Polygon POINTS="{cut_to_poly(cut)}"/></Shape>\n'
                        f'              </Glyph>'
                    )
                strings_xml.append(
                    f'            <String ID="segment_{w_idx}"\n'
                    f'                    CONTENT="{e(wb["word"])}"\n'
                    f'                    HPOS="{wb["x1"]}" VPOS="{wb["y1"]}"\n'
                    f'                    WIDTH="{w_w}" HEIGHT="{w_h}"\n'
                    f'                    WC="1.0000">\n'
                    + '\n'.join(glyphs_xml) + '\n'
                    + '            </String>'
                )
            content_xml = '\n'.join(strings_xml)
        else:
            content_xml = (
                f'            <String CONTENT="{e(gt_text)}"\n'
                f'                    HPOS="{line["hpos"]}" VPOS="{line["vpos"]}"\n'
                f'                    WIDTH="{line["width"]}" HEIGHT="{line["height"]}"\n'
                f'                    ></String>'
            )

        tagrefs = f'\n                    TAGREFS="{line["tags"]}"' if line['tags'] else ''
        lines_xml.append(
            f'          <TextLine ID="{line["id"]}"{tagrefs}\n'
            f'                    BASELINE="{bl_str}"\n'
            f'                    HPOS="{line["hpos"]}" VPOS="{line["vpos"]}"\n'
            f'                    WIDTH="{line["width"]}" HEIGHT="{line["height"]}">\n'
            f'            <Shape><Polygon POINTS="{poly_str}"/></Shape>\n'
            + content_xml + '\n'
            + '          </TextLine>'
        )

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<alto xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
        '      xmlns="http://www.loc.gov/standards/alto/ns-v4#"\n'
        '      xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v4#'
        ' http://www.loc.gov/standards/alto/v4/alto-4-2.xsd">\n'
        '  <Description>\n'
        '    <MeasurementUnit>pixel</MeasurementUnit>\n'
        '    <sourceImageInformation>\n'
        f'      <fileName>{e(img_name)}</fileName>\n'
        '    </sourceImageInformation>\n'
        '  </Description>\n'
        '  <Layout>\n'
        f'    <Page WIDTH="{page_w}" HEIGHT="{page_h}"\n'
        '          PHYSICAL_IMG_NR="0" ID="eSc_dummypage_">\n'
        f'      <PrintSpace HPOS="0" VPOS="0" WIDTH="{page_w}" HEIGHT="{page_h}">\n'
        '        <TextBlock ID="eSc_dummyblock_">\n'
        + '\n'.join(lines_xml) + '\n'
        + '        </TextBlock>\n'
          '      </PrintSpace>\n'
          '    </Page>\n'
          '  </Layout>\n'
          '</alto>'
    )


# =============================================================================
# 6. PIPELINE PRINCIPAL
# =============================================================================

_net_cache = {}

def _load_model(model_path: str):
    if model_path not in _net_cache:
        _net_cache[model_path] = models.load_any(model_path)
    return _net_cache[model_path]


def process_file(xml_path: str, img_path: str, model_path: str,
                 output_path: str, verbose: bool = False):
    """Traite un fichier ALTO + image et produit un ALTO word-level."""
    print(f"  {Path(xml_path).name}", end='', flush=True)

    try:
        lines, page_w, page_h, img_name, skipped = parse_alto(xml_path)
    except ET.ParseError as e:
        print(f"\n  {err(f'[!] XML invalide : {e}')}")
        return

    if skipped:
        print(f" {warn(f'({skipped} lignes sans baseline skippées)')}", end='')

    if not lines:
        print(f"\n  {warn('[!] Aucune ligne à traiter')}")
        return

    img = Image.open(img_path)
    net = _load_model(model_path)

    recognized = recognize_lines(img, lines, net)

    n_failed = 0
    for rec in recognized:
        if verbose:
            print(f"\n    GT  : '{rec['gt_text']}'")
            print(f"    OCR : '{rec['ocr_text']}'")
        rec['word_bboxes'] = build_word_bboxes(
            rec['gt_text'], rec['ocr_text'], rec['cuts']
        )
        if not rec['word_bboxes']:
            n_failed += 1
        if verbose and rec['word_bboxes']:
            for wb in rec['word_bboxes']:
                print(f"      '{wb['word']}' -> ({wb['x1']},{wb['y1']},{wb['x2']},{wb['y2']})")

    alto_xml = build_alto_xml(recognized, page_w, page_h, img_name)
    Path(output_path).write_text(alto_xml, encoding='utf-8')
    n_words = sum(len(r['word_bboxes']) for r in recognized)

    if n_failed == 0:
        status = ok(f"-> {len(recognized)} lignes, {n_words} mots")
    elif n_failed < len(recognized):
        status = warn(f"-> {len(recognized)} lignes, {n_words} mots ({n_failed} sans positions)")
    else:
        status = err(f"-> {len(recognized)} lignes, aucune position générée")

    print(f"\n  {status} -> {Path(output_path).name}")


def main():
    parser = argparse.ArgumentParser(
        description='Alignement positions Kraken + texte GT pour ALTO word-level',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Fichier unique (écrase l'original)
  python alto_wordlevel.py --xml data/page.xml --img data/page.jpg --model models/model.mlmodel

  # Fichier unique avec sortie séparée
  python alto_wordlevel.py --xml data/page.xml --img data/page.jpg --model models/model.mlmodel --output out.xml

  # Dossier entier (écrase les originaux)
  python alto_wordlevel.py --xml_dir data/ --model models/model.mlmodel

  # Dossier entier avec sortie séparée
  python alto_wordlevel.py --xml_dir data/ --output_dir data_wl/ --model models/model.mlmodel
        """
    )
    parser.add_argument('--xml',        help='Fichier ALTO en entrée (mode fichier unique)')
    parser.add_argument('--img',        help='Image correspondante (mode fichier unique)')
    parser.add_argument('--xml_dir',    help='Dossier contenant les ALTO (mode dossier)')
    parser.add_argument('--img_dir',    help='Dossier images (défaut = xml_dir)')
    parser.add_argument('--model',      required=True, help='Modèle Kraken (.mlmodel)')
    parser.add_argument('--output',     help='Fichier de sortie (défaut = écrase l\'original)')
    parser.add_argument('--output_dir', help='Dossier de sortie (défaut = écrase les originaux)')
    parser.add_argument('--verbose',    action='store_true', help='Afficher les détails ligne par ligne')
    args = parser.parse_args()

    if args.xml and args.img:
        output = args.output or args.xml
        process_file(args.xml, args.img, args.model, output, args.verbose)

    elif args.xml_dir:
        xml_dir = Path(args.xml_dir)
        img_dir = Path(args.img_dir) if args.img_dir else xml_dir
        out_dir = Path(args.output_dir) if args.output_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        # Backup des XML originaux avant modification
        if not out_dir:
            import zipfile, datetime
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = xml_dir.parent / f"{xml_dir.name}_backup_{stamp}.zip"
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for xml_path in xml_dir.glob("*.xml"):
                    zf.write(xml_path, xml_path.name)
            print(f"Backup : {backup_path} ({len(list(xml_dir.glob('*.xml')))} fichiers)")

        xml_files = sorted(xml_dir.glob('*.xml'))
        print(f"Traitement de {len(xml_files)} fichiers...")

        for xml_path in xml_files:
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                p = img_dir / xml_path.with_suffix(ext).name
                if p.exists():
                    img_path = p
                    break
            if img_path is None:
                print(f"  {err(f'[!] Image non trouvée pour {xml_path.name}')}")
                continue

            out_path = (out_dir / xml_path.name) if out_dir else xml_path
            try:
                process_file(str(xml_path), str(img_path), args.model,
                             str(out_path), args.verbose)
            except Exception as e:
                print(f"  {err(f'[!] Erreur sur {xml_path.name} : {e}')}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
