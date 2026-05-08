import gradio as gr
import joblib
import json
import numpy as np
import os

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model     = joblib.load(os.path.join(BASE_DIR, "models", "best_model.pkl"))

with open(os.path.join(BASE_DIR, "models", "model_info.json")) as f:
    model_info = json.load(f)

FEATURES  = ["protein","fat","carbohydrate","fiber","sugars","sodium",
             "saturated_fatty_acids","monounsaturated_fatty_acids","polyunsaturated_fatty_acids"]

def calorie_level(cal):
    if cal < 100:   return "🟢 Low Calorie"
    elif cal < 250: return "🟡 Moderate"
    elif cal < 450: return "🟠 High Calorie"
    else:           return "🔴 Very High Calorie"

def predict(protein, fat, carbohydrate, fiber, sugars,
            sodium, saturated_fat, mono_fat, poly_fat):
    values = [protein, fat, carbohydrate, fiber, sugars,
              sodium, saturated_fat, mono_fat, poly_fat]
    arr    = np.array(values).reshape(1, -1)
    cal    = max(0, round(float(model.predict(arr)[0]), 1))
    lvl    = calorie_level(cal)

    # Atwater formula check
    atwater = round(protein*4 + fat*9 + carbohydrate*4, 1)

    result = f"""
## {lvl}
# 🔥 {cal} kcal / 100g

---

## 📊 Your Input Summary

| Nutrient | Amount | Calories |
|---|---|---|
| 🥩 Protein | {protein:.1f} g | {protein*4:.0f} kcal |
| 🧈 Fat | {fat:.1f} g | {fat*9:.0f} kcal |
| 🍞 Carbohydrate | {carbohydrate:.1f} g | {carbohydrate*4:.0f} kcal |
| 🌾 Fiber | {fiber:.1f} g | — |
| 🍬 Sugars | {sugars:.1f} g | — |
| 🧂 Sodium | {sodium:.0f} mg | — |
| Saturated Fat | {saturated_fat:.1f} g | — |
| Mono Fat | {mono_fat:.1f} g | — |
| Poly Fat | {poly_fat:.1f} g | — |

---

## 🧮 Formula Check
| | |
|---|---|
| Atwater Formula | {atwater} kcal |
| **ML Model Predicted** | **{cal} kcal** |
| Difference | {abs(cal - atwater):.1f} kcal |

---
*{model_info["best_model"]} · R² = {model_info["r2_score"]} · MAE = {model_info["mae"]} kcal*
"""
    return result

EXAMPLES = [
    [31.0, 3.6,  0.0,  0.0,  0.0,  74,  1.0, 1.2, 0.8],
    [0.3,  0.2, 26.9,  2.4, 14.4,   1,  0.0, 0.0, 0.1],
    [21.2, 49.9, 21.6, 12.5,  4.4,  1,  3.8,31.6,12.1],
    [6.9,  4.5,  0.7,  0.0,  0.0,  56,  1.3, 1.4, 0.6],
    [2.7,  0.3, 28.2,  0.4,  0.0,   1,  0.1, 0.1, 0.1],
    [4.6, 17.8, 18.6,  1.2,  3.4, 550,  6.4, 8.2, 2.0],
]

EXAMPLE_LABELS = [
    "🍗 Chicken Breast",
    "🍎 Apple",
    "🥜 Almonds",
    "🥚 Egg",
    "🍚 White Rice",
    "🍕 Pizza",
]

with gr.Blocks(
    title="🍕 Food Calorie Estimator",
    theme=gr.themes.Soft(primary_hue="orange")
) as demo:

    gr.Markdown("""
# 🍕 Food Calorie Estimator
**Enter the nutritional values of any food → ML model predicts its calories & breaks them down**

> Model: **{model}** · R² = **{r2}** · Average error = only **{mae} kcal**
""".format(model=model_info["best_model"], r2=model_info["r2_score"], mae=model_info["mae"]))

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔢 Main Macronutrients (per 100g)")
            protein      = gr.Slider(0, 100, value=10,  step=0.1, label="🥩 Protein (g)")
            fat          = gr.Slider(0, 100, value=5,   step=0.1, label="🧈 Fat (g)")
            carbohydrate = gr.Slider(0, 100, value=20,  step=0.1, label="🍞 Carbohydrate (g)")

            gr.Markdown("### 🌿 Other Nutrients")
            fiber        = gr.Slider(0, 50,   value=2,  step=0.1, label="🌾 Fiber (g)")
            sugars       = gr.Slider(0, 100,  value=5,  step=0.1, label="🍬 Sugars (g)")
            sodium       = gr.Slider(0, 5000, value=50, step=1,   label="🧂 Sodium (mg)")

            gr.Markdown("### 🫧 Fatty Acids (g)")
            saturated    = gr.Slider(0, 50, value=1, step=0.1, label="Saturated Fat (g)")
            mono         = gr.Slider(0, 50, value=1, step=0.1, label="Monounsaturated Fat (g)")
            poly         = gr.Slider(0, 50, value=1, step=0.1, label="Polyunsaturated Fat (g)")

            btn = gr.Button("🔍 Predict Calories", variant="primary", size="lg")

        with gr.Column(scale=1):
            out = gr.Markdown("*Adjust the sliders and click Predict Calories ↑*")

    btn.click(
        fn=predict,
        inputs=[protein, fat, carbohydrate, fiber, sugars, sodium, saturated, mono, poly],
        outputs=[out]
    )

    gr.Markdown("### 💡 Quick Examples — click any food to auto-fill:")
    gr.Examples(
        examples=EXAMPLES,
        inputs=[protein, fat, carbohydrate, fiber, sugars, sodium, saturated, mono, poly],
        example_labels=EXAMPLE_LABELS,
        fn=predict,
        outputs=[out],
        cache_examples=False
    )

    gr.Markdown("""
---
**Dataset:** USDA FoodData Central (8,600+ foods) |
**Author:** Abdulkader Tamer — NTI Creativa 2025
""")

if __name__ == "__main__":
    demo.launch()
