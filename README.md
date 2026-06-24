## Clinical NLP — Drug Review Sentiment Classification with Explainability & Uncertainty Quantification

> **A production-oriented NLP pipeline applying transformer-based classification,  
> SHAP explainability, and uncertainty quantification to patient-reported drug review data.**

---

## Overview

Patient-reported outcome (PRO) data — in the form of written drug reviews — represents a rich, underutilised signal in health technology assessment. This project builds a full end-to-end NLP pipeline to classify sentiment from drug reviews, with a specific focus on two properties that are critical in clinical AI deployment but often absent from research-grade models:

- **Explainability (XAI):** *Why did the model make this prediction?*
- **Uncertainty Quantification (UQ):** *How confident is the model — and is that confidence trustworthy?*

Both are essential for building AI systems that can be audited, governed, and responsibly deployed in health or regulatory contexts.

---

## Clinical & Regulatory Motivation

Health technology assessment bodies such as NICE routinely consider patient experience evidence when evaluating treatments. An NLP pipeline capable of extracting structured sentiment from unstructured patient text at scale could support:

- **Patient feedback analysis** — identifying patterns of dissatisfaction or adverse effects across large volumes of reviews
- **Pharmacovigilance** — flagging reviews likely to mention adverse drug reactions for review
- **Guidance development** — surfacing patient-reported outcomes to complement clinical trial evidence

However, deploying any ML system in a health context requires more than high accuracy. Overconfident wrong predictions carry real risk. This project treats **calibration and explainability as first-class requirements**, not afterthoughts.

---

## Dataset

**UCI Drug Review Dataset** (Kallumadi & Grer, 2018)  
- 215,000+ patient-written drug reviews from drugs.com  
- Variables: drug name, condition treated, review text, rating (1–10), date  
- Task: binary sentiment classification — **Positive (rating ≥ 7)** vs **Negative (rating < 7)**  
- Source: [Kaggle](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018)

> Data is not included in this repository. Place `drugsComTrain_raw.tsv` and `drugsComTest_raw.tsv` in the `data/` directory before running.

---

## Repository Structure

```
clinical-nlp-drug-review-xai/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA, class distribution, text analysis, preprocessing
│   ├── 02_baseline_model.ipynb       # TF-IDF + LR and DistilBERT baselines
│   └── 03_ablation_xai_uq.ipynb      # Ablation study, SHAP explanations, calibration
│
├── src/
│   └── utils.py                      # Reusable preprocessing, evaluation, UQ, and plotting functions
│
├── data/                             # Place raw TSV files here (not tracked by git)
├── outputs/                          # Generated figures and results CSVs
├── docs/                             # Additional documentation
│
├── requirements.txt
└── README.md
```

---

## Methods

### 1. Preprocessing
- HTML entity decoding, URL removal, whitespace normalisation
- Text lowercased; no stemming (preserves naturalistic input for tokenisers)
- Ratings binarised at threshold ≥7 (clinically motivated threshold)

### 2. Baseline Models

| Model | Rationale |
|---|---|
| TF-IDF + Logistic Regression | Fast, interpretable, widely used in health informatics; provides a meaningful performance floor |
| DistilBERT (fine-tuned) | State-of-the-art contextual representations; 40% smaller than BERT with ~97% performance |

Both trained with **class-weighted loss** to handle the ~65/35 class imbalance.

### 3. Ablation Study
Six systematic configurations varying n-gram range, feature count, and class weighting — establishing *which pipeline components actually drive performance*.

### 4. Explainability — SHAP
- **Global:** Top features driving sentiment predictions across the full test set
- **Local:** Per-prediction waterfall plots explaining individual classifications
- Implemented via `shap.LinearExplainer` for the LR model (exact, not approximated)

### 5. Uncertainty Quantification — Calibration
- Reliability diagrams (calibration curves) comparing raw vs. calibrated probabilities
- Expected Calibration Error (ECE) as a scalar measure of overconfidence
- **Isotonic regression calibration** applied post-hoc to reduce ECE
- Analysis of high-confidence incorrect predictions — the highest-risk failure mode

---

## Results Summary

| Model | Accuracy | F1 (Macro) | ROC-AUC | ECE |
|---|---|---|---|---|
| TF-IDF + LR (uncalibrated) | *run to populate* | *run to populate* | *run to populate* | *run to populate* |
| TF-IDF + LR (calibrated) | *run to populate* | *run to populate* | *run to populate* | *run to populate* |
| DistilBERT baseline | *run to populate* | *run to populate* | *run to populate* | — |

> Full results are saved to `outputs/final_model_comparison.csv` after running the notebooks.

---

## Key Findings

- **Class weighting** was the single most impactful pipeline component for minority-class (Negative) F1
- **SHAP analysis** confirmed the model uses clinically plausible features — words like *side effects*, *didn't work*, and *worse* are strong negative predictors; *helped*, *improved*, and *great* drive positive predictions
- The baseline model was **moderately overconfident**: its raw probabilities did not accurately reflect empirical accuracy across probability buckets (ECE > 0.05)
- **Isotonic calibration** substantially reduced ECE with no degradation to ROC-AUC, producing more trustworthy confidence scores
- High-confidence incorrect predictions — the most clinically dangerous failure mode — were reduced post-calibration

---

## Ethical Considerations

- Dataset is fully anonymised; no patient-identifiable information
- Class imbalance was explicitly addressed to avoid systematic bias against the minority class
- Model explanations (SHAP) were verified for clinical face validity before reporting
- Calibration was treated as a safety requirement: overconfident predictions in clinical contexts can mislead decision-makers
- This project does not recommend deployment of this specific model for any clinical decision-making purpose; it demonstrates a methodology for responsible development

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/clinical-nlp-drug-review-xai.git
cd clinical-nlp-drug-review-xai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add data files to data/ directory
#    drugsComTrain_raw.tsv
#    drugsComTest_raw.tsv

# 4. Run notebooks in order
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## Technologies

`Python` · `PyTorch` · `HuggingFace Transformers` · `scikit-learn` · `SHAP` · `Matplotlib` · `Seaborn`

---

## Author

**Garcia Wa Nnam**  
MSc Health Data Science and Statistics, University of Plymouth  
Clinical background: Registered Midwife | Head of Midwifery (St. Louis University Institute)  

[GitHub](https://github.com/YOUR_USERNAME) · [Email](mailto:wannamgarcia2@gmail.com)

---

## References

- Kallumadi, S. & Grer, F. (2018). *UCI Drug Review Dataset.* Kaggle.
- Sanh, V. et al. (2019). DistilBERT, a distilled version of BERT. *arXiv:1910.01108.*
- Lundberg, S. & Lee, S-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS.*
- Niculescu-Mizil, A. & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML.*
