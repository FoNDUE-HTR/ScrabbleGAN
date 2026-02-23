# ScrabbleGAN for French

A fine-tuning and generation pipeline for [ScrabbleGAN](https://github.com/arshjot/ScrabbleGAN) (arshjot's implementation) applied to historical French handwriting (HTR). Includes ALTO word-level alignment via Kraken, legacy weight conversion, and synthetic data generation compatible with eScriptorium and Kraken.

## Overview

This pipeline builds on [arshjot/ScrabbleGAN](https://github.com/arshjot/ScrabbleGAN).

```
ALTO/image pairs
    │
    ├── alto_wordlevel.py     →  word-level ALTO (Kraken positions + GT text preserved)
    │
    └── scrabblegan_pipeline.py
            ├── finetune      →  fine-tune ScrabbleGAN on your word patches
            └── generate      →  synthetic line images + ALTO v4
```

---

## Scripts

| Script | Description |
|--------|-------------|
| `alto_wordlevel.py` | Converts eScriptorium line-level ALTO to word-level using Kraken alignment |
| `scrabblegan_pipeline.py` | Fine-tunes ScrabbleGAN and generates synthetic handwriting images |
| `convert_weights.py` | Converts legacy PyTorch weights (`.pkl` + blob folder) to modern `.pt` format |

---

## Requirements

- Python 3.10+
- macOS or Linux (CPU or MPS/CUDA)

```bash
pip install -r requirements.txt
```

## Pre-requisites

Weights can be found attache to the [va release](https://github.com/FoNDUE-HTR/ScrabbleGAN/releases/tag/v1).

```
models/
  fondue_archimed_v4.mlmodel   # Kraken HTR model
  RIMES_char_map.pkl           # 93-character French alphabet mapping
  RIMES_data_reshaped.pt       # ScrabbleGAN RIMES checkpoint (converted)
Lexique383.tsv                 # French lexicon
data/
  page_001.xml                 # ALTO eScriptorium
  page_001.jpg                 # paired image
  ...
```

---

## Data Format

ALTO XML files paired with their images, same base name:

```
data/
  page_001.xml
  page_001.jpg
  page_002.xml
  page_002.png
  ...
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`

### ALTO format note

Two ALTO variants are handled:

- **eScriptorium line-level**: one `<String>` per `<TextLine>` with the full line — produced by eScriptorium by default
- **Word-level**: one `<String>` per word with individual bounding boxes — needed for fine-tuning and richer training data

Use `alto_wordlevel.py` to convert line-level to word-level before fine-tuning.

---

## Script 1 — alto_wordlevel.py

Converts eScriptorium line-level ALTO to word-level ALTO by:

1. Running Kraken recognition on each line to get per-character positions (cuts)
2. Aligning OCR output with GT text via Levenshtein to recover correct positions even when OCR makes mistakes
3. Reconstructing word bounding boxes from the aligned character cuts
4. Writing a new ALTO v4 with `<String>` per word and `<Glyph>` per character

Lines already at word-level (multiple `<String>` per `<TextLine>`) are automatically skipped. The original `fileName` from the ALTO metadata is preserved in the output.

### Usage

```bash
# Single file — overwrites the original
python alto_wordlevel.py \
    --xml data/page.xml \
    --img data/page.jpg \
    --model models/fondue_archimed_v4.mlmodel

# Single file — separate output
python alto_wordlevel.py \
    --xml data/page.xml \
    --img data/page.jpg \
    --model models/fondue_archimed_v4.mlmodel \
    --output data/page_out.xml

# Entire folder — overwrites originals (creates a timestamped zip backup first)
python alto_wordlevel.py \
    --xml_dir data/ \
    --model models/fondue_archimed_v4.mlmodel

# Entire folder — separate output folder
python alto_wordlevel.py \
    --xml_dir data/ \
    --output_dir data_wordlevel/ \
    --model models/fondue_archimed_v4.mlmodel
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--xml` | — | Input ALTO file (single file mode) |
| `--img` | — | Paired image (single file mode) |
| `--xml_dir` | — | Folder containing ALTO files (batch mode) |
| `--img_dir` | same as `xml_dir` | Folder containing images (batch mode) |
| `--model` | required | Kraken HTR model (`.mlmodel`) |
| `--output` | overwrites original | Output file (single file mode) |
| `--output_dir` | overwrites originals | Output folder (batch mode) |
| `--verbose` | off | Show GT/OCR/position details per line |

### Notes

- When overwriting originals in batch mode, a timestamped zip backup is created automatically (`data_backup_YYYYMMDD_HHMMSS.zip`)
- Errors (invalid XML, missing image, failed line) are always reported regardless of `--verbose`
- Lines where OCR error rate is high will still be processed but word positions may be approximate — use `--verbose` to inspect
- Works best with a Kraken model trained on the same script type

---

## Script 2 — scrabblegan_pipeline.py

### Step 1 — Fine-tuning

```bash
python scrabblegan_pipeline.py --step finetune \
    --weights ./models/RIMES_data_reshaped.pt \
    --charmap ./models/RIMES_char_map.pkl \
    --alto_dir ./data \
    --output_dir ./synthetic \
    --epochs 100
```

This step:
1. Loads the RIMES character map (93 French characters)
2. Extracts word patches from all ALTO `<String>` elements, resized to H=32px
3. Formats the dataset as `{'word_data': {id: (label, img_array)}, 'char_map': {...}}`
4. Copies the checkpoint to `scrabblegan_arshjot/weights/model_checkpoint_epoch_0.pth.tar`
5. Rewrites `config.py` with absolute paths, launches `train.py`, then restores `config.py`
6. Saves the best checkpoint (lowest `R_real`) to `scrabblegan_arshjot/weights/model_best.pth.tar`

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--weights` | required | Path to converted RIMES checkpoint (`.pt`) |
| `--charmap` | required | Path to `RIMES_char_map.pkl` |
| `--alto_dir` | `./data` | Folder containing ALTO/image pairs |
| `--output_dir` | `./synthetic_v2` | Output folder |
| `--epochs` | `10` | Number of training epochs (100 recommended) |
| `--save_patches` | off | Save extracted word patches as image+txt pairs in the given folder |

**Key metrics:**

| Metric | Meaning |
|--------|---------|
| `R_real` | Recognizer loss on real images — should decrease. Target: < 1.0 |
| `R_fake` | Recognizer loss on generated images — should stay low |
| `G` | Generator loss |
| `D` | Discriminator loss |

---

### Step 2 — Generate synthetic images

```bash
python scrabblegan_pipeline.py --step generate \
    --weights scrabblegan_arshjot/weights/model_best.pth.tar \
    --charmap ./models/RIMES_char_map.pkl \
    --alto_dir ./data \
    --output_dir ./synthetic \
    --n_images 5
```

This step:
1. Extracts all unique line texts from your ALTO files
2. Splits each line into words, generates each word separately
3. Assembles words horizontally with cross-fade blending and colour-matched gaps
4. Saves each image as `.png` with a matching `.txt` and `.xml` (ALTO v4 word-level)
5. Writes `manifest_generated.csv` ready for Kraken/Calamari

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--weights` | required | Fine-tuned checkpoint (use `model_best.pth.tar`) |
| `--charmap` | required | Path to `RIMES_char_map.pkl` |
| `--alto_dir` | `./data` | Folder with ALTO files |
| `--output_dir` | `./synthetic_v2_ft` | Output folder |
| `--n_images` | `5` | Synthetic images per line |
| `--sharpen` | off | Apply 2-pass sharpening + ×2 upscale to generated images |

---

## Script 3 — convert_weights.py

Converts legacy PyTorch ScrabbleGAN weights to modern `.pt` format.

```bash
# Convert
python convert_weights.py \
    --pkl ./data.pkl \
    --data_dir ./data \
    --output ./models/RIMES_data_reshaped.pt

# Verify
python convert_weights.py --verify ./models/RIMES_data_reshaped.pt
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--pkl` | `./data.pkl` | Legacy pickle file |
| `--data_dir` | `./data` | Folder containing tensor blobs |
| `--output` | `./data_final.pt` | Output `.pt` file |
| `--verify` | — | Verify an existing `.pt` file |

---

## Output Structure

```
synthetic_v2_ft/
  generated/
    00001_Cellules_nerveuses_000.png
    00001_Cellules_nerveuses_000.txt   # plain text transcription
    00001_Cellules_nerveuses_000.xml   # ALTO v4 word-level
    00001_Cellules_nerveuses_001.png
    ...
  manifest_generated.csv               # image_path \t text

scrabblegan_arshjot/weights/
  model_checkpoint_epoch_1.pth.tar
  ...
  model_best.pth.tar                   # best checkpoint (lowest R_real)
```

---

## Using the output with Kraken

```bash
ketos train -f alto -d ./synthetic_v2_ft/manifest_generated.csv
```

Combine real and synthetic data:

```bash
cat synthetic_v2/manifest_patches.csv synthetic_v2_ft/manifest_generated.csv > manifest_all.csv
ketos train -f alto -d ./manifest_all.csv
```

---

## Notes

- **macOS MPS**: `pin_memory` warning is harmless
- **Batch size**: default is 8. With word-level ALTO you get ~4-5× more patches than line-level
- **Epochs**: 100 epochs ≈ 65 minutes on CPU (Apple Silicon). `R_real` typically converges around epoch 50-70
- **Best model**: `model_best.pth.tar` is automatically saved whenever `R_real` improves
- **Character filtering**: characters outside the 93-character RIMES alphabet are removed rather than discarded
- **alto_wordlevel.py**: works best with a model trained on the same script type; a high OCR error rate means word positions may be approximate for that line

## License

This pipeline is released under the Apache 2.0 License. It builds on [arshjot/ScrabbleGAN](https://github.com/arshjot/ScrabbleGAN), itself based on [ScrabbleGAN](https://github.com/amzn/style-based-gan-pytorch) by Amazon. Please refer to their respective licenses.

## Contact

[Simon Gabay](https://www.unige.ch/lettres/humanites-numeriques/equipe/collaborateurs/simon-gabay)
