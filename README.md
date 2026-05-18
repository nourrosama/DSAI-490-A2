# DSAI 490 — Assignment 2: Conditional Date Generator

Four generative models trained to produce calendar dates that satisfy a set of conditions: day of week, month, leap year flag, and decade.

---

## Task

Given a condition like `[MON] [DEC] [False] [196]`, generate a valid date that satisfies all four constraints — in this case, a Monday in December of a non-leap year in the 1960s.

**Input format** (one condition per line):
```
[MON] [DEC] [False] [196]
[SAT] [JAN] [True] [200]
```

**Output format** (condition + generated date):
```
[MON] [DEC] [False] [196] 3-12-1962
[SAT] [JAN] [True] [200] 20-1-2004
```

---

## Models

| # | Model | Type | Category |
|---|-------|------|----------|
| 1 | CVAE | Conditional Variational Autoencoder | In-course |
| 2 | CGAN | Conditional GAN with Gumbel-Softmax | In-course |
| 3 | SeqGAN | Sequence GAN with REINFORCE | Outside-course |
| 4 | D3PM | Discrete Absorbing Diffusion | Outside-course |

---

## Repository Structure

```
DSAI-490-A2/
├── data/
│   ├── data.txt               # Full dataset (~146k labelled dates)
│   └── example_input.txt      # 1465 conditions for evaluation
│
├── model/
│   ├── tokenizer.py           # Hybrid tokenizer (vocab_size=35)
│   ├── dataset.py             # DateDataset + make_loaders()
│   ├── metrics.py             # CSR, validity, diversity evaluation
│   ├── predict.py             # Unified inference script (all 4 models)
│   │
│   ├── vae/
│   │   ├── model.py           # CVAE architecture
│   │   ├── train.py           # ELBO training loop
│   │   └── weights/           # Saved checkpoints + history
│   │
│   ├── cgan/
│   │   ├── model.py           # Generator + Discriminator (Gumbel-Softmax)
│   │   ├── train.py           # GAN training loop
│   │   └── weights/
│   │
│   ├── seqgan/
│   │   ├── model.py           # LSTM Generator + BiGRU Discriminator
│   │   ├── train.py           # 3-phase training (MLE pretrain → D pretrain → REINFORCE)
│   │   └── weights/
│   │
│   └── d3pm/
│       ├── model.py           # Absorbing diffusion + CFG
│       ├── train.py           # Hybrid VLB + CE loss + guidance sweep
│       └── weights/
│
├── DSAI490_A2_Training.ipynb  # Full Colab training notebook
├── model_comparison.png       # Side-by-side metric bar chart
├── environment.yml            # Conda environment (CPU)
└── .gitignore
```

---

## Tokenizer

A hybrid tokenizer with a fixed vocabulary of 35 tokens:

| Token type | Coverage | IDs |
|------------|----------|-----|
| Special (PAD, BOS, EOS) | `[PAD]`, `[BOS]`, `[EOS]` | 0–2 |
| Day of week | `[MON]`–`[SUN]` | 3–9 |
| Month | `[JAN]`–`[DEC]` | 10–21 |
| Leap flag | `[False]`, `[True]` | 22–23 |
| Digits | `0`–`9` | 24–33 |
| Hyphen | `-` | 34 |

Conditions encode to 6 tokens (day + month + leap + 3 decade digits). Dates encode character-by-character using digit and hyphen tokens.

---

## Inference

Run any of the four models from the `model/` directory:

```bash
cd model/

# CVAE
python predict.py -i ../data/example_input.txt -o output.txt --model vae

# CGAN
python predict.py -i ../data/example_input.txt -o output.txt --model cgan --temperature 0.8

# SeqGAN
python predict.py -i ../data/example_input.txt -o output.txt --model seqgan

# D3PM
python predict.py -i ../data/example_input.txt -o output.txt --model d3pm --guidance 2.0
```

The script automatically loads `*_best.pt` if available, otherwise `*_last.pt`.

---

## Training

Each model has its own training script. All scripts accept `--data`, `--epochs`, and `--batch` arguments.

```bash
# VAE
cd model/vae && python train.py --epochs 60 --batch 512

# CGAN
cd model/cgan && python train.py --epochs 100 --batch 512 --temperature 0.8

# SeqGAN
cd model/seqgan && python train.py --pretrain_g 5 --pretrain_d 1 --adv_epochs 25 --batch 512

# D3PM
cd model/d3pm && python train.py --T 50 --epochs 30 --batch 512 --guidance 2.0
```

For GPU training, use the provided Colab notebook `DSAI490_A2_Training.ipynb`.

---

## Evaluation

```python
from metrics import evaluate, print_metrics

with open("model/vae/weights/predictions.txt") as f:
    preds = [l.strip() for l in f if l.strip()]

print_metrics(evaluate(preds), prefix="VAE")
```

**Metrics:**

| Metric | Description |
|--------|-------------|
| Overall CSR | All 4 conditions satisfied simultaneously |
| Day CSR | Day-of-week matches |
| Month CSR | Month matches |
| Leap CSR | Leap year flag matches |
| Decade CSR | Decade (first 3 year digits) matches |
| Validity Rate | Generated date is a real calendar date |
| Diversity | Unique valid dates / total valid dates |

---

## Results

| Model | Overall CSR | Validity | Diversity |
|-------|-------------|----------|-----------|
| VAE | ~14% | ~95% | ~0.95 |
| CGAN | ~13% | ~97% | ~0.76 |
| SeqGAN | ~14% | ~97% | ~0.95 |
| D3PM | ~15% | ~96% | ~0.87 |

All models achieve ~99% on Month, Leap, and Decade CSR. The bottleneck is Day CSR — the day-of-week is deterministic given a specific date, so the model must generate a full date that happens to land on the correct weekday.

D3PM supports classifier-free guidance. At w=2 (default), it achieves the best balance of CSR and validity. At w=4, validity drops sharply to ~44%.

---

## Environment

```bash
conda env create -f environment.yml
conda activate dsai490-a2
```

Requires Python 3.10+, PyTorch 2.x. For GPU training, replace `cpuonly` with the appropriate CUDA channel in `environment.yml`.
