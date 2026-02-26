"""
pairs_to_alto.py - Convertit des paires NOM.txt + NOM.(jpg|png|tif|jpeg) en ALTO v4
=====================================================================================

Pour chaque paire trouvée dans le dossier source :
    - NOM.txt  → contient le texte transcrit (une ligne)
    - NOM.jpg  → (ou .png, .tif, .jpeg) image correspondante

Produit un fichier ALTO v4 par paire : NOM.xml

La baseline est placée à 20 % en partant du bas (soit VPOS * 0.80).

Usage :
    python pairs_to_alto.py --input_dir paires/ --output_dir alto_out/

Options :
    --input_dir   Dossier contenant les paires .txt / image
    --output_dir  Dossier de sortie pour les fichiers ALTO (.xml)
    --encoding    Encodage des fichiers .txt (défaut : utf-8)
"""

import argparse
from pathlib import Path
from html import escape as e
from PIL import Image


IMAGE_EXTS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']


def read_text(txt_path: Path, encoding: str) -> str:
    """Lit le contenu d'un fichier texte et retourne la première ligne non vide."""
    text = txt_path.read_text(encoding=encoding).strip()
    # Garde uniquement la première ligne si le fichier en contient plusieurs
    return text.splitlines()[0].strip() if text else ''


def make_alto(stem: str, text: str, img_path: Path) -> str:
    img = Image.open(img_path)
    w, h = img.size
    # Baseline à 20 % depuis le bas → y = h * 0.80
    baseline_y = int(h * 0.80)
    filename = img_path.name

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<alto xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
        '      xmlns="http://www.loc.gov/standards/alto/ns-v4#"\n'
        '      xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v4#'
        ' http://www.loc.gov/standards/alto/v4/alto-4-2.xsd">\n'
        '  <Description>\n'
        '    <MeasurementUnit>pixel</MeasurementUnit>\n'
        '    <sourceImageInformation>\n'
        f'      <fileName>{e(filename)}</fileName>\n'
        '    </sourceImageInformation>\n'
        '  </Description>\n'
        '  <Layout>\n'
        f'    <Page WIDTH="{w}" HEIGHT="{h}"\n'
        '          PHYSICAL_IMG_NR="0" ID="eSc_dummypage_">\n'
        f'      <PrintSpace HPOS="0" VPOS="0" WIDTH="{w}" HEIGHT="{h}">\n'
        '        <TextBlock ID="eSc_dummyblock_">\n'
        f'          <TextLine ID="{e(stem)}"\n'
        f'                    BASELINE="0 {baseline_y} {w} {baseline_y}"\n'
        f'                    HPOS="0" VPOS="0" WIDTH="{w}" HEIGHT="{h}">\n'
        f'            <Shape><Polygon POINTS="0 0 {w} 0 {w} {h} 0 {h}"/></Shape>\n'
        f'            <String CONTENT="{e(text)}"\n'
        f'                    HPOS="0" VPOS="0" WIDTH="{w}" HEIGHT="{h}"/>\n'
        '          </TextLine>\n'
        '        </TextBlock>\n'
        '      </PrintSpace>\n'
        '    </Page>\n'
        '  </Layout>\n'
        '</alto>'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Convertit des paires NOM.txt + NOM.image en ALTO v4'
    )
    parser.add_argument('--input_dir',  required=True, help='Dossier contenant les paires txt/image')
    parser.add_argument('--output_dir', required=True, help='Dossier de sortie pour les ALTO')
    parser.add_argument('--encoding',   default='utf-8', help='Encodage des fichiers .txt (défaut : utf-8)')
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Recense tous les fichiers .txt du dossier source
    txt_files = sorted(input_dir.glob('*.txt'))
    print(f"Fichiers .txt trouvés : {len(txt_files)}")

    ok = skipped = errors = 0

    for txt_path in txt_files:
        stem = txt_path.stem

        # Cherche l'image correspondante
        img_path = None
        for ext in IMAGE_EXTS:
            candidate = input_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"  [!] Image non trouvée pour : {stem}")
            skipped += 1
            continue

        try:
            text = read_text(txt_path, args.encoding)
            if not text:
                print(f"  [!] Texte vide, ignoré : {stem}")
                skipped += 1
                continue

            alto = make_alto(stem, text, img_path)
            out_path = output_dir / f"{stem}.xml"
            out_path.write_text(alto, encoding='utf-8')
            ok += 1

        except Exception as ex:
            print(f"  [!] Erreur sur {stem} : {ex}")
            errors += 1

    print(f"\nRésultat : {ok} ALTO générés, {skipped} ignorés, {errors} erreurs")
    print(f"Sortie : {output_dir}")


if __name__ == '__main__':
    main()
