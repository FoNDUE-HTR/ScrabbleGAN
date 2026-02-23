"""
normalize_alto.py - Filtre les caractères hors char_map dans les attributs CONTENT des ALTO
==========================================================================================

Retire (ou remplace) tout caractère absent du char_map dans les attributs CONTENT="..."
des fichiers ALTO. Utile pour rendre les données compatibles avec un modèle ScrabbleGAN
(IAM 74 caractères, RIMES 93 caractères, etc.).

Les diacritiques sont d'abord normalisés (é→e, à→a, etc.) avant filtrage, ce qui
maximise la conservation du texte.

Usage :
    # IAM
    python normalize_alto.py --xml_dir alto_out/ --charmap models/IAM_char_map.pkl \\
        --output_dir alto_iam/ --report

    # RIMES
    python normalize_alto.py --xml_dir alto_out/ --charmap models/RIMES_char_map.pkl \\
        --output_dir alto_rimes/ --report
"""

import argparse
import pickle
import re
import unicodedata
from pathlib import Path


def load_charmap(charmap_path: str) -> set:
    """Charge un char_map pickle et retourne l'ensemble des caractères valides."""
    with open(charmap_path, 'rb') as f:
        cm = pickle.load(f, encoding='latin1')
    if isinstance(cm, dict):
        chars = set(cm.keys())
    else:
        chars = set(cm)
    # Nettoyer : garder seulement les chaînes d'un caractère
    chars = {str(c) for c in chars if len(str(c)) == 1}
    # Toujours garder l'espace
    chars.add(' ')
    return chars


def strip_diacritics(text: str) -> str:
    """Retire les diacritiques : é→e, à→a, ç→c, etc."""
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')


def filter_text(text: str, valid_chars: set, report_changes: list = None) -> str:
    """
    Filtre un texte pour ne garder que les caractères valides.
    Pour chaque caractère :
      1. Décoder les entités XML (&quot; → ", &amp; → &, etc.)
      2. Si le caractère est dans le char_map → garder tel quel
      3. Sinon tenter la normalisation NFD (é→e) → garder si dans char_map
      4. Sinon retirer
    Ré-encode les entités XML et nettoie les espaces multiples.
    """
    from html import unescape, escape as xml_escape

    # Étape 1 : décoder les entités XML et normaliser en NFC
    decoded = unicodedata.normalize('NFC', unescape(text))

    # Étape 2-3 : filtrer caractère par caractère
    result = []
    for c in decoded:
        if c in valid_chars:
            # Le caractère est valide tel quel (y compris les accents du char_map)
            result.append(c)
        else:
            # Tenter la normalisation NFD (retire le diacritique)
            fallback = strip_diacritics(c)
            if fallback and fallback in valid_chars:
                result.append(fallback)
            # sinon on retire silencieusement

    # Étape 4 : nettoyer les espaces multiples
    filtered = re.sub(r' +', ' ', ''.join(result)).strip()

    # Étape 5 : ré-encoder pour XML
    filtered_encoded = xml_escape(filtered, quote=True)

    if report_changes is not None and text != filtered_encoded:
        report_changes.append((text, filtered_encoded))

    return filtered_encoded


def process_file(xml_path: Path, output_path: Path,
                 valid_chars: set, report: bool = False):
    content = xml_path.read_text(encoding='utf-8')
    changes = []

    def replace_content(m):
        original = m.group(1)
        filtered = filter_text(original, valid_chars,
                               report_changes=changes if report else None)
        return f'CONTENT="{filtered}"'

    result = re.sub(r'CONTENT="([^"]*)"', replace_content, content)
    output_path.write_text(result, encoding='utf-8')

    if report and changes:
        print(f"  {xml_path.name}")
        for before, after in changes:
            print(f"    {before!r} -> {after!r}")


def main():
    parser = argparse.ArgumentParser(
        description='Filtre les caractères hors char_map dans les ALTO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # IAM
  python normalize_alto.py --xml_dir alto_out/ --charmap models/IAM_char_map.pkl \\
      --output_dir alto_iam/ --report

  # RIMES
  python normalize_alto.py --xml_dir alto_out/ --charmap models/RIMES_char_map.pkl \\
      --output_dir alto_rimes/ --report
        """
    )
    parser.add_argument('--xml_dir',    required=True, help='Dossier contenant les ALTO')
    parser.add_argument('--charmap',    required=True, help='Fichier char_map .pkl (IAM, RIMES...)')
    parser.add_argument('--output_dir', default=None,  help='Dossier de sortie (défaut = écrase les originaux)')
    parser.add_argument('--report',     action='store_true', help='Afficher les changements effectués')
    args = parser.parse_args()

    valid_chars = load_charmap(args.charmap)
    print(f"Char_map chargé : {len(valid_chars)} caractères valides")
    if args.report:
        print(f"  {sorted(valid_chars)}")

    xml_dir    = Path(args.xml_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(xml_dir.glob('*.xml'))
    print(f"Traitement de {len(xml_files)} fichiers...")

    for xml_path in xml_files:
        out_path = (output_dir / xml_path.name) if output_dir else xml_path
        try:
            process_file(xml_path, out_path, valid_chars, report=args.report)
        except Exception as ex:
            print(f"  [!] {xml_path.name} : {ex}")

    print(f"-> {len(xml_files)} fichiers traités")


if __name__ == '__main__':
    main()
