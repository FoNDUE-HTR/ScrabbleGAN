# HTR Pipeline — Synthetic Data & Word-Level Alignment

A pipeline for generating synthetic handwriting training data and enriching ALTO annotations with word-level bounding boxes, using ScrabbleGAN pre-trained on RIMES (French handwriting). Output is compatible with [Kraken](https://kraken.re), [Calamari](https://github.com/Calamari-OCR/calamari), and [eScriptorium](https://escriptorium.fr).

## Overview

```
ALTO/image pairs
    │
    ├── align_alto.py  →  word-level ALTO (positions from Kraken + GT text preserved)
    │
    └── htr_synth_v2.py
            ├── finetune  →  fine-tune ScrabbleGAN on your word patches
            └── generate  →  synthetic line images + ALTO v4
```

---

## Requirements

- Python 3.10+
- macOS or Linux (CPU or MPS/CUDA)

```bash
pip install torch torchvision pillow numpy tensorboard pandas kraken
```

## Pre-requisites

The following files must be present:

```
models/
  fondue_archimed_v4.mlmodel   # Kraken HTR model
  RIMES_char_map.pkl           # 93-character French alphabet mapping
  data_final.pt                # ScrabbleGAN RIMES checkpoint (converted)
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

- **eScriptorium line-level**: one `<String>` per `<TextLine>` with the full line text — produced by eScriptorium by default
- **Word-level**: one `<String>` per word with individual bounding boxes — needed for ScrabbleGAN fine-tuning and richer training data

`align_alto.py` converts line-level ALTO to word-level. `htr_synth_v2.py` works best with word-level input.

---

## Script 1 — align_alto.py

Converts eScriptorium line-level ALTO to word-level ALTO by:

1. Running Kraken recognition on each line to get per-character positions (cuts)
2. Aligning the OCR output with the GT text via Levenshtein to recover correct positions even when OCR makes mistakes
3. Reconstructing word bounding boxes from the aligned character cuts
4. Writing a new ALTO v4 with `<String>` per word and `<Glyph>` per character

Lines already at word-level (multiple `<String>` per `<TextLine>`) are automatically skipped.

### Usage

```bash
# Single file — overwrites the original
python align_alto.py \
    --xml data/page.xml \
    --img data/page.jpg \
    --model models/fondue_archimed_v4.mlmodel

# Single file — separate output
python align_alto.py \
    --xml data/page.xml \
    --img data/page.jpg \
    --model models/fondue_archimed_v4.mlmodel \
    --output data/page_out.xml

# Entire folder — overwrites originals
python align_alto.py \
    --xml_dir data/ \
    --model models/fondue_archimed_v4.mlmodel

# Entire folder — separate output folder
python align_alto.py \
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

- Errors (invalid XML, missing image, failed line) are always reported regardless of `--verbose`
- Lines where OCR error rate is high will still be processed but word positions may be approximate — use `--verbose` to inspect
- The `fileName` in the output ALTO is taken from the original ALTO metadata, not from the XML filename

---

## Script 2 — htr_synth_v2.py

### Step 1 — Fine-tuning

```bash
python htr_synth_v2.py --step finetune \
    --weights ./models/data_final.pt \
    --charmap ./models/RIMES_char_map.pkl \
    --alto_dir ./data \
    --output_dir ./synthetic_v2 \
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
| `--weights` | required | Path to `data_final.pt` (converted RIMES checkpoint) |
| `--charmap` | required | Path to `RIMES_char_map.pkl` |
| `--alto_dir` | `./data` | Folder containing ALTO/image pairs |
| `--output_dir` | `./synthetic_v2` | Output folder |
| `--epochs` | `10` | Number of training epochs (100 recommended) |
| `--save_patches` | off | Save extracted word patches as image+txt pairs in the given folder (useful to verify bounding boxes) |

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
python htr_synth_v2.py --step generate \
    --weights scrabblegan_arshjot/weights/model_best.pth.tar \
    --charmap ./models/RIMES_char_map.pkl \
    --alto_dir ./data \
    --output_dir ./synthetic_v2_ft \
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

## Output Structure

```
synthetic_v2_ft/
  generated/
    00001_Histologie_000.png
    00001_Histologie_000.txt   # plain text transcription
    00001_Histologie_000.xml   # ALTO v4 word-level
    00001_Histologie_001.png
    ...
  manifest_generated.csv       # image_path \t text

scrabblegan_arshjot/weights/
  model_checkpoint_epoch_1.pth.tar
  ...
  model_best.pth.tar           # best checkpoint (lowest R_real)
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
- **Character filtering**: characters outside the 93-character RIMES alphabet are removed (not discarded)
- **One-time patches to ScrabbleGAN**: `training_utils.py` and `ScrabbleGAN.py` require manual patches (optimizer loading tolerance, lexicon loading) — see commit history
- **align_alto.py**: works best with a model trained on the same script type. A high OCR error rate on a line means word positions may be approximate for that line
