# ScrabbleGAN for French

A fine-tuning and generation pipeline for [ScrabbleGAN](https://github.com/arshjot/ScrabbleGAN) (arshjot's implementation) applied to historical French handwriting (HTR). Includes ALTO word-level alignment via Kraken, legacy weight conversion, and synthetic data generation compatible with eScriptorium and Kraken.

## Overview

This pipeline builds on [arshjot/ScrabbleGAN](https://github.com/arshjot/ScrabbleGAN).

```
ALTO/image pairs
    │
    ├── --step contrast    →  (optional) gamma correction on images before training
    │
    ├── --step wordlevel   →  word-level ALTO (Kraken positions + GT text preserved)
    │                         images copied to output folder
    │
    ├── --step normalize   →  normalize ALTO text to char_map (diacritics + filtering)
    │                         images copied to output folder
    │
    ├── --step finetune    →  fine-tune ScrabbleGAN on your word patches
    │                         previous checkpoints archived (timestamped)
    │
    ├── --step generate    →  synthetic line images + ALTO v4
    │
    └── --step style       →  (optional) apply real document colours (IAM / RIMES preset)
```

---

## Scripts

| Step | Description |
|------|-------------|
| `--step contrast` | *(optional)* Gamma correction on images before fine-tuning |
| `--step wordlevel` | Converts eScriptorium line-level ALTO to word-level using Kraken alignment |
| `--step normalize` | Normalizes ALTO text to a char_map (diacritics, unsupported characters) |
| `--step finetune` | Fine-tunes ScrabbleGAN on your word patches |
| `--step generate` | Generates synthetic handwriting images + ALTO v4 |
| `--step style` | *(optional)* Applies real document colours to synthetic images |

All steps are invoked via a single script: `scrabblegan_pipeline.py`

Legacy utility: `convert_weights.py` — converts legacy PyTorch weights (`.pkl` + blob folder) to modern `.pt` format

---

## Requirements

- Python 3.10+
- macOS or Linux (CPU or MPS/CUDA)

```bash
pip install -r requirements.txt
```

## Pre-requisites

Weights can be found attached to the [v1 release](https://github.com/FoNDUE-HTR/ScrabbleGAN/releases/tag/v1).

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

## Step 0 — contrast *(optional)*

Applies gamma correction to all images in a folder before fine-tuning. This can improve the GAN's ability to learn contrast between ink and background.

- `gamma < 1` : darkens the image (useful if the background is too light)
- `gamma > 1` : lightens the image (useful if the image is too dark overall)

XML (ALTO) files and images are automatically copied to the output folder to preserve pairs.

### Usage

```bash
python scrabblegan_pipeline.py --step contrast --input_dir data/ --output_dir data_gamma/ --gamma 0.8
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input_dir` | required | Folder containing images (and paired XML files) |
| `--output_dir` | required | Output folder |
| `--gamma` | `0.8` | Gamma value (`<1` darkens, `>1` lightens) |

---

## Step 1 — wordlevel

Converts eScriptorium line-level ALTO to word-level ALTO by:

1. Running Kraken recognition on each line to get per-character positions (cuts)
2. Aligning OCR output with GT text via Levenshtein to recover correct positions even when OCR makes mistakes
3. Reconstructing word bounding boxes from the aligned character cuts
4. Writing a new ALTO v4 with `<String>` per word and `<Glyph>` per character

Lines already at word-level are automatically reprocessed to integrate annotator corrections. The original `fileName` from the ALTO metadata is preserved in the output.

### Usage

```bash
# Entire folder — separate output folder (images are copied automatically)
python scrabblegan_pipeline.py --step wordlevel --xml_dir data/ --output_dir data_wordlevel/ --model models/fondue_archimed_v4.mlmodel

# Overwrite originals (creates a timestamped zip backup first)
python scrabblegan_pipeline.py --step wordlevel --xml_dir data/ --model models/fondue_archimed_v4.mlmodel
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--xml_dir` | required | Folder containing ALTO files |
| `--img_dir` | same as `xml_dir` | Folder containing images |
| `--model` | required | Kraken HTR model (`.mlmodel`) |
| `--output_dir` | overwrites originals | Output folder (images are copied here automatically) |
| `--verbose` | off | Show GT/OCR/position details per line |

### Notes

- When overwriting originals in batch mode, a timestamped zip backup is created automatically (`data_backup_YYYYMMDD_HHMMSS.zip`)
- Errors (invalid XML, missing image, failed line) are always reported regardless of `--verbose`
- Lines where OCR error rate is high will still be processed but word positions may be approximate — use `--verbose` to inspect
- Works best with a Kraken model trained on the same script type

---

## Step 2 — normalize

Normalizes the text content of ALTO files to match a given char_map, for compatibility with ScrabbleGAN (RIMES or IAM). For each character in each `CONTENT` attribute:

1. If the character is in the char_map → kept as-is (accented characters are preserved when present in the char_map)
2. If not, a diacritic normalization is attempted (é→e) — kept if the result is in the char_map
3. Otherwise removed

XML entities (`&amp;`, `&quot;`, etc.) are decoded before processing and re-encoded after. Unicode is normalized to NFC before comparison.

> ⚠️ This step is only relevant when **fine-tuning an existing model** (RIMES or IAM). If you are training from scratch with your own char_map, skip this step — normalizing to a pre-existing alphabet makes no sense in that context.

### Usage

```bash
# RIMES char_map — preserves French accents
python scrabblegan_pipeline.py --step normalize --xml_dir data_wordlevel/ --charmap models/RIMES_char_map.pkl --output_dir data_normalized/ --report

# IAM char_map — strips all accents (é→e, à→a, no French characters)
python scrabblegan_pipeline.py --step normalize --xml_dir data_wordlevel/ --charmap models/IAM_char_map.pkl --output_dir data_normalized_iam/ --report

# Remove additional characters not in the char_map
python scrabblegan_pipeline.py --step normalize --xml_dir data_wordlevel/ --charmap models/RIMES_char_map.pkl --output_dir data_normalized/ --addchar models/supplementary_out_characters.txt --report
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--xml_dir` | required | Folder containing ALTO files |
| `--charmap` | required | Char_map pickle file (RIMES, IAM, or any ScrabbleGAN `.pkl`) |
| `--output_dir` | overwrites originals | Output folder (images are copied here automatically) |
| `--addchar` | — | Text file (1 char/line) listing characters to remove from the char_map |
| `--report` | off | Print before/after changes per file |

### Notes

- With RIMES: accented characters (é, à, ç…) are preserved; only truly unsupported characters (`:`, `;`, `.`, `,`…) are removed
- With IAM: all accents are stripped (é→e, à→a); the 74-character IAM alphabet has no French-specific characters
- Use `--report` first to inspect changes before committing to an output folder

---

## Steps 3–4 — finetune & generate

### Step 3 — finetune

```bash
python scrabblegan_pipeline.py --step finetune --weights ./models/RIMES_data_reshaped.pt --charmap ./models/RIMES_char_map.pkl --alto_dir ./data_normalized/ --output_dir ./synthetic --epochs 100

# Save output to log while monitoring in real time
python scrabblegan_pipeline.py --step finetune --weights ./models/RIMES_data_reshaped.pt --charmap ./models/RIMES_char_map.pkl --alto_dir ./data_normalized/ --output_dir ./synthetic --epochs 100 2>&1 | tee train_RIMES.log
```

This step:
1. Loads the RIMES character map (93 French characters)
2. Extracts word patches from all ALTO `<String>` elements, resized to H=32px
3. Formats the dataset as `{'word_data': {id: (label, img_array)}, 'char_map': {...}}`
4. Archives any existing checkpoints in a timestamped subfolder (`weights/archive_YYYYMMDD_HHMMSS/`)
5. Copies the new checkpoint to `scrabblegan_arshjot/weights/model_checkpoint_epoch_0.pth.tar`
6. Rewrites `config.py` with absolute paths, launches `train.py`, then restores `config.py`
7. Saves the best checkpoint (lowest `R_real`) to `scrabblegan_arshjot/weights/model_best.pth.tar`

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

### Step 4 — generate

```bash
python scrabblegan_pipeline.py --step generate --weights scrabblegan_arshjot/weights/model_best.pth.tar --charmap ./models/RIMES_char_map.pkl --alto_dir ./data_normalized/ --output_dir ./synthetic --n_images 5
```

> ⚠️ The `--charmap` and `--weights` must match: a RIMES checkpoint (93 chars) requires `RIMES_char_map.pkl`, an IAM checkpoint (74 chars) requires `IAM_char_map.pkl`.

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


---

## Utility — convert_weights.py

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

## Step 5 — style *(optional)*

Applies the colour palette of real historical documents to synthetic ScrabbleGAN images, making them visually more realistic. Uses colour sampling (ink + background) from real images rather than texture blending, to avoid halos and texture artefacts.

Two presets are available depending on your source corpus:

| Config | Best for | ink_darken | mask_gamma |
|--------|----------|------------|------------|
| `IAM` | Normal contrast documents | 0.5 | 1.5 |
| `RIMES` | Low contrast / medical documents | 0.15 | 2.5 |

### Usage

```bash
# Single image
python scrabblegan_pipeline.py --step style --synth_dir synthetic/generated/ --real_dir backgrounds/ --output_dir synthetic_styled/ --config IAM

# Low-contrast / medical documents
python scrabblegan_pipeline.py --step style --synth_dir synthetic/generated/ --real_dir backgrounds/ --output_dir synthetic_styled/ --config RIMES
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--synth_dir` | required | Folder of synthetic images |
| `--real_dir` | required | Folder of real document images (background samples) |
| `--output_dir` | `<synth_dir>_styled/` | Output folder |
| `--config` | `IAM` | Colour preset: `IAM` or `RIMES` |


### Notes

- `.txt` transcription files are automatically copied to the output folder
- The `--real_dir` folder should contain representative patches of the target document style (background texture, colour)

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
  archive_20260225_143022/             # previous session checkpoints (auto-archived)
    model_best.pth.tar
    ...
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

- **macOS MPS**: `pin_memory` warning is harmless; `ctc_loss` falls back to CPU automatically via `PYTORCH_ENABLE_MPS_FALLBACK=1`
- **Batch size**: default is 8. With word-level ALTO you get ~4-5× more patches than line-level
- **Epochs**: 100 epochs ≈ 65 minutes on CPU (Apple Silicon). `R_real` typically converges around epoch 50-70
- **Best model**: `model_best.pth.tar` is automatically saved whenever `R_real` improves
- **Character filtering**: run `--step normalize` before fine-tuning to ensure text labels and image content are in sync
- **wordlevel**: works best with a Kraken model trained on the same script type; a high OCR error rate means word positions may be approximate for that line
- **Checkpoint/charmap match**: always use the same charmap for `finetune` and `generate` — a RIMES checkpoint (93 chars) is incompatible with an IAM charmap (74 chars) and will raise a size mismatch error

## License

This pipeline is released under the Apache 2.0 License. It builds on [arshjot/ScrabbleGAN](https://github.com/arshjot/ScrabbleGAN), itself based on [ScrabbleGAN](https://github.com/amzn/style-based-gan-pytorch) by Amazon. Please refer to their respective licenses.

## Contact

[Simon Gabay](https://www.unige.ch/lettres/humanites-numeriques/equipe/collaborateurs/simon-gabay)
