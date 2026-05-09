---
title: 🍕 Food Calorie Estimator
emoji: 🍕
colorFrom: orange
colorTo: yellow
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
short_description: Photo → AI identifies food → ML predicts calories (R²=0.9947)
---

# 🍕 Food Calorie Estimator

> **Snap a photo of any food → get its calories + full macronutrient breakdown.**
> A two-stage AI pipeline that combines OpenAI's CLIP for vision with a
> scikit-learn calorie regressor trained on 8,789 USDA foods.

[![Hugging Face Spaces](https://img.shields.io/badge/🤗_Spaces-live_demo-yellow?style=flat-square)](https://huggingface.co/spaces/YOUR_USERNAME/food-calorie-estimator)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)](https://www.python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?style=flat-square)](https://scikit-learn.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat-square)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-5.x-orange?style=flat-square)](https://gradio.app)
[![R²](https://img.shields.io/badge/R²-0.9947-brightgreen?style=flat-square)]()
[![MAE](https://img.shields.io/badge/MAE-6.42_kcal-brightgreen?style=flat-square)]()

---

## ✨ What it does

1. You upload a photo of food.
2. **CLIP** (zero-shot vision) embeds the image and finds the most similar
   food among 230 curated prompts.
3. The corresponding USDA row is looked up.
4. A **Linear Regression** model trained on 8,789 foods predicts calories
   from the 9 macronutrients.
5. You see calories, a full nutritional table, the macro breakdown
   (Atwater factors), and the model's confidence.

---

## 🧠 The two AI models

| Stage | Model | Why this one |
|-------|-------|--------------|
| 👁️ Vision | `openai/clip-vit-base-patch32` | Zero-shot — recognizes any food we describe in plain English; no retraining when the catalog grows |
| 🧮 Calories | `LinearRegression` (scikit-learn) | Beat Random Forest, Decision Tree, Gradient Boosting — calories really *are* a linear function of macros (Atwater) |

### Why CLIP, not a fixed classifier?

The previous version used `nateraw/food` (Food-101) — only 101 labels,
mostly restaurant dishes (no apple, no rice, no chicken breast). CLIP's
text encoder turns *any* English food description into a vector, which
means we can grow the catalog from 230 → 1,000 foods just by editing a
list — no fine-tuning, no GPU.

---

## 📊 ML pipeline (per the NTI Creativa task)

| Stage | Notebook | What's inside |
|-------|----------|---------------|
| 1 | `01_data_acquisition.ipynb` | Pull USDA dataset from Kaggle |
| 2 | `02_preprocessing_eda.ipynb` | Clean strings, impute medians, EDA |
| 3 | `03_model_training.ipynb` | 4 models, 5-fold CV, GridSearch tuning |
| 4 | `04_evaluation_results.ipynb` | Feature importance, residuals, recommendation |

### Model comparison (per Notebook 03)

| Model | R² | MAE (kcal) | CV Mean | CV Std |
|-------|----|-----------|---------|--------|
| **Linear Regression** | **0.9947** | **6.42** | **0.9907** | 0.0022 |
| Gradient Boosting     | 0.9941 | 6.53 | 0.9911 | 0.0017 |
| Random Forest (Tuned) | 0.9915 | 7.10 | 0.9902 | 0.0015 |
| Random Forest         | 0.9911 | 8.40 | 0.9894 | 0.0016 |
| Decision Tree         | 0.9822 | 13.79| 0.9784 | 0.0032 |

### Why the simplest model won

Calories follow the **Atwater General Factors** almost exactly:

```
calories ≈ 4·protein + 4·carbohydrate + 9·fat
```

Linear Regression rediscovers these coefficients directly from data,
which is why no ensemble can beat it. Lesson: when the underlying
process is linear, a fancier model just adds variance.

---

## 🗂️ Project structure

```
food-calorie-estimator/
├── app.py                       # Gradio app (HF Spaces entry-point)
├── foods_catalog.py             # 230 CLIP prompts + USDA search terms
├── requirements.txt             # ~8 pinned dependencies
├── README.md                    # this file (with HF metadata header)
│
├── notebooks/                   # The 4-step ML pipeline
│   ├── 01_data_acquisition.ipynb
│   ├── 02_preprocessing_eda.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation_results.ipynb
│
├── data/processed/
│   └── nutrition_clean.csv      # 8,789 cleaned USDA foods (committed)
│
├── models/
│   ├── best_model.pkl           # 705 B trained Linear Regression
│   ├── scaler.pkl               # 1.1 KB StandardScaler
│   └── model_info.json          # metadata
│
├── results/
│   ├── plots/                   # generated EDA + evaluation figures
│   └── metrics/model_comparison.csv
│
└── src/                         # plain-Python ports of the notebooks
    ├── preprocess.py
    ├── train.py
    └── evaluate.py
```

---

## 🚀 Run locally

```bash
git clone https://github.com/YOUR_USERNAME/food-calorie-estimator
cd food-calorie-estimator

python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

python app.py                       # opens http://localhost:7860
```

The first run downloads ~150 MB of CLIP weights from Hugging Face Hub,
caches them under `~/.cache/huggingface`, then computes 230 text
embeddings and stores them in `models/clip_text_embeddings.npy`. Every
subsequent launch is instant.

### Run the notebooks

```bash
jupyter lab notebooks/
```

---

## 🌐 Deploy your own copy on Hugging Face Spaces

1. Create a new Space → SDK: **Gradio** → hardware: **CPU basic (free)**.
2. Push this repo to the Space's Git remote (Spaces is a Git host).
3. The metadata block at the top of this README tells Spaces to use
   `app.py` as the entry-point — no extra config needed.
4. First boot is ~2 min (downloading CLIP); subsequent boots are <30 s.

The whole stack — CLIP + Linear Regression + 230 catalog foods — fits
in well under the free-tier 16 GB / 2 vCPU limits.

---

## 🔬 What I learned from this project

- **Always start with a baseline.** Linear Regression beating Random
  Forest taught me to never assume "more complex = better" — when the
  signal is linear, a complex model just memorizes noise.
- **Cross-validation matters.** A single 80/20 split gave R² = 0.9947;
  5-fold CV gave 0.9907 ± 0.0022 — the gap is the actual generalization
  estimate I'd report to a stakeholder.
- **Feature scaling has to follow the model.** Linear Regression was
  trained on `StandardScaler`-ed inputs, so the deployed app *must*
  apply the saved scaler before predicting — otherwise the coefficients
  point at the wrong feature scale.
- **Data leakage is sneaky.** I `fit` the scaler on `X_train` only, then
  `transform` both train and test — fitting on the full data would have
  let the test set peek at training-time statistics.
- **Zero-shot vision changes the deployment story.** Replacing Food-101
  (101 labels, fixed) with CLIP (any label, free-text) turned a 1-day
  retraining loop into a one-line catalog edit.

---

## 🛠️ Tech stack

| Layer | Library |
|-------|---------|
| Vision | PyTorch · transformers · CLIP |
| ML pipeline | scikit-learn · pandas · numpy |
| Notebooks | Jupyter Lab · matplotlib · seaborn |
| App | Gradio |
| Deployment | Hugging Face Spaces · GitHub |

---

## 📊 Dataset

- **Source:** [Kaggle — Nutritional Values for Common Foods](https://www.kaggle.com/datasets/trolukovich/nutritional-values-for-common-foods-and-products)
- **Origin:** USDA FoodData Central (public domain)
- **Size:** 8,789 cleaned foods × 9 macronutrient features

---

## 👤 Author

**Abdulkader Tamer** — AI Engineering Student, Mansoura National University
Built for **NTI Training Project** — Machine Learning track.

## 📄 License

MIT — see [LICENSE](LICENSE).
