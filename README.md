# HMM Viterbi POS Tagger

A Python implementation of a **Hidden Markov Model (HMM)** Part-of-Speech (POS) tagger with **Viterbi decoding**, including training, evaluation, data analysis, figure generation, and an auto-generated PDF report.

---

## Project Structure

```
HMM_Viterbi/
├── data/
│   ├── train.txt          # POS-tagged training corpus (word/TAG format)
│   └── test.txt           # POS-tagged test corpus
├── figures/               # Auto-generated analysis figures (PNG)
├── report/
│   ├── __init__.py
│   ├── generate_report.py # PDF report generator
│   └── report.pdf         # Generated PDF report (after running main.py)
├── src/
│   ├── __init__.py
│   ├── hmm_model.py       # HMM model (MLE + Laplace smoothing)
│   ├── viterbi.py         # Viterbi decoding algorithm
│   ├── train.py           # Data loading and training
│   ├── evaluate.py        # Evaluation metrics (accuracy, F1, confusion matrix)
│   └── visualize.py       # Figure generation (heat-maps, bar charts)
├── tests/
│   └── test_hmm_viterbi.py # Unit tests
├── main.py                # Entry point – runs the full pipeline
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Requirements

- Python 3.9 or later
- Required libraries (install with pip):

```bash
pip install -r requirements.txt
```

The dependencies are:

| Package      | Purpose                               |
|--------------|---------------------------------------|
| `matplotlib` | Plotting figures                      |
| `numpy`      | Numerical operations                  |
| `seaborn`    | Statistical visualisation             |
| `reportlab`  | PDF report generation                 |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourexpress/HMM_Viterbi.git
cd HMM_Viterbi
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
python main.py
```

This single command will:
1. **Train** the HMM on `data/train.txt`
2. **Evaluate** on `data/test.txt` and print accuracy / per-tag F1
3. **Generate figures** in the `figures/` directory
4. **Create a PDF report** at `report/report.pdf`

### 4. View the report

Open `report/report.pdf` with any PDF viewer.

---

## Command-Line Options

```
python main.py [--train TRAIN_FILE] [--test TEST_FILE]
               [--figures FIGURES_DIR] [--report REPORT_FILE]
               [--smoothing SMOOTHING]

Options:
  --train      Path to the training corpus   (default: data/train.txt)
  --test       Path to the test corpus       (default: data/test.txt)
  --figures    Output directory for figures  (default: figures/)
  --report     Output path for PDF report    (default: report/report.pdf)
  --smoothing  Laplace smoothing constant    (default: 1.0)
```

**Example – use custom files and stronger smoothing:**

```bash
python main.py --train data/train.txt --test data/test.txt --smoothing 0.1
```

---

## Data Format

Each corpus file contains one sentence per line.  Each token is
`word/TAG` separated by spaces.  Penn Treebank-style tags are used.

```
The/DT cat/NN sat/VBD on/IN the/DT mat/NN ./.
She/PRP loves/VBZ reading/VBG books/NNS ./.
```

Supported tags include: `CC`, `CD`, `DT`, `IN`, `JJ`, `JJR`, `NN`,
`NNP`, `NNS`, `PRP`, `PRP$`, `RB`, `RP`, `TO`, `VB`, `VBD`, `VBG`,
`VBN`, `VBP`, `VBZ`, `.`

---

## Algorithm Overview

### Hidden Markov Model

A first-order HMM models a sentence as:

```
P(w₁:ₙ, t₁:ₙ) = P(t₁|START) × ∏ P(tᵢ|tᵢ₋₁) × ∏ P(wᵢ|tᵢ)
```

Parameters are estimated by maximum-likelihood with **Laplace (add-one)
smoothing** to handle unseen events and out-of-vocabulary words.

### Viterbi Decoding

The Viterbi algorithm finds the most-probable tag sequence in
*O(n · |T|²)* time using dynamic programming in log-space:

```
δₜ(j) = max_i [ δₜ₋₁(i) + log A(i,j) ] + log B(j, wₜ)
```

Back-pointers are stored at each step so the optimal path can be
recovered via a final back-trace pass.

---

## Generated Figures

| File                       | Description                                    |
|----------------------------|------------------------------------------------|
| `confusion_matrix.png`     | Row-normalised confusion matrix heat-map       |
| `per_tag_f1.png`           | Per-tag F1 bar chart with accuracy reference   |
| `precision_recall_f1.png`  | Grouped bar chart: precision, recall, F1       |
| `transition_matrix.png`    | HMM transition probability heat-map           |
| `top_emissions.png`        | Top-8 emission words per tag                   |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Report Contents

The auto-generated PDF (`report/report.pdf`) includes:

1. Executive Summary
2. Project Workflow
3. Dataset Description and Tag Set
4. Model Description (HMM parameters, smoothing, Viterbi algorithm)
5. Evaluation Results (overall accuracy, per-tag precision/recall/F1)
6. Analysis Figures (all five charts embedded)
7. Discussion (error analysis, linguistic observations)
8. Conclusions and recommended extensions

---

## License

This project is released under the MIT License – see [LICENSE](LICENSE) for details.
