import os
import cv2
import numpy as np
import tensorflow as tf
import shap
import random
import matplotlib.pyplot as plt
from transformers import pipeline
from io import BytesIO
import base64

# === CONFIG ===
MODEL_PATH = "/mnt/d/IIT/Degree/4th year/Final Year Project/Tire detection System/TireVisionPro/InceptionResNetV2_tire_defect_model_TPU_2.keras"
TEST_IMAGE_PATH = "/mnt/d/IIT/Degree/4th year/Final Year Project/Tire detection System/TireVisionPro/Tyre Classification Gray Scale/train/Good/good-668-_jpg.rf.23011e261d8c60deebe0daad53c68c97.jpg"
BACKGROUND_DIR = "/mnt/d/IIT/Degree/4th year/Final Year Project/Tire detection System/TireVisionPro/Tyre Classification Gray Scale/test/Good"

# === Load Model ===
model = tf.keras.models.load_model(MODEL_PATH)

# === Load and Preprocess Target Image ===
img = cv2.imread(TEST_IMAGE_PATH)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (299, 299))
img_array = np.expand_dims(img_resized / 255.0, axis=0)

# === Prediction ===
prediction = model.predict(img_array)[0][0]
is_defective = prediction < 0.3
print(f"\n✅ Prediction Score: {prediction:.6f}")
print("🚨 Defective Tire Detected! ⚠️" if is_defective else "✅ Tire is in Good Condition!")

# === Load 10 Background Images ===
background_images = []
all_files = [f for f in os.listdir(BACKGROUND_DIR) if f.lower().endswith((".jpg", ".png"))]
random.shuffle(all_files)

for fname in all_files[:200]:
    path = os.path.join(BACKGROUND_DIR, fname)
    bg_img = cv2.imread(path)
    if bg_img is not None:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        bg_img = cv2.resize(bg_img, (299, 299))
        bg_img = bg_img / 255.0
        background_images.append(bg_img)

background_array = np.array(background_images)

# === SHAP (GradientExplainer) ===
with tf.device("/CPU:0"):
    explainer = shap.GradientExplainer((model.input, model.output), background_array)
    shap_values = explainer.shap_values(img_array)


# === SHAP Processing ===
shap_array = shap_values[0][0]  # shape: (299, 299, 3)
heatmap_raw = np.sum(np.abs(shap_array), axis=-1)

# === Normalize and Create Overlay ===
heatmap_norm = np.uint8(255 * heatmap_raw / (np.max(heatmap_raw) + 1e-8))
heatmap_blurred = cv2.GaussianBlur(heatmap_norm, (5, 5), 0)
heatmap_colored = cv2.applyColorMap(heatmap_blurred, cv2.COLORMAP_JET)
heatmap_colored = cv2.resize(heatmap_colored, (img_resized.shape[1], img_resized.shape[0]))
overlay = cv2.addWeighted(img_resized, 0.6, heatmap_colored, 0.4, 0)

# === SHAP Stats ===
print("\n🔍 SHAP Value Summary:")
print("Shape:", heatmap_raw.shape)
print("Min SHAP value:", np.min(heatmap_raw))
print("Max SHAP value:", np.max(heatmap_raw))
print("Mean SHAP value:", np.mean(heatmap_raw))
print("Sum of Absolute SHAP values:", np.sum(np.abs(heatmap_raw)))
print("Non-zero elements:", np.count_nonzero(heatmap_raw))

# === LLM Explanation ===
generator = pipeline("text2text-generation", model="facebook/bart-large-cnn", device=-1)


# ✅ SHAP Explanation LLM Generator (Highly Detailed)
def generate_shap_explanation_detailed(prediction_score, is_defective, shap_max, shap_mean):
    explanation_prompt = f"""
    The AI-powered tire defect detection system used **SHAP (SHapley Additive exPlanations)** to explain how the model made its decision.

    🔍 **What is SHAP?**
    SHAP is an Explainable AI method that assigns each pixel a contribution score — showing how much each region of the tire image influenced the final prediction.
    High SHAP values mean those parts played an important role in the model’s decision.

    ✅ **Prediction Details:**
    - Prediction Score: {prediction_score:.4f}
    - Tire Classified As: {"Defective" if is_defective else "Good"}
    - SHAP Contribution Stats: Max SHAP = {shap_max:.6f}, Mean SHAP = {shap_mean:.6f}

    ### 📊 SHAP-Based Explanation:
    1️⃣ **Important Pixel Regions:** Areas with high SHAP values contributed most to the decision.
    2️⃣ **Model Confidence:** SHAP distribution indicates the model’s certainty — higher mean suggests confident prediction.
    3️⃣ **Transparency:** SHAP revealed the model focused on relevant features (like treads, texture, wear spots).
    4️⃣ **Final Verdict:** The tire is classified as {"DEFECTIVE" if is_defective else "GOOD"} based on the evidence.

    ### ✅ Recommended Action:"""

    if is_defective:
        recommendations = "- 🚨 **Immediate replacement advised.** SHAP highlights potential defects like cracks, bulges, or tread issues.\n" \
                          "- 🔍 **Manual inspection strongly recommended.**\n" \
                          "- 🛑 **Do not use tire for long-distance travel.**"
    else:
        recommendations = "- ✅ **Tire appears healthy and structurally sound.**\n" \
                          "- 🔍 **SHAP confirms model’s low-risk decision.**\n" \
                          "- 🧰 **Continue regular checks for safety and maintenance.**"

    full_prompt = explanation_prompt + "\n" + recommendations

    # 🧠 Call BART model with better generation settings
    response = generator(full_prompt, max_new_tokens=1000, do_sample=True, temperature=0.8)

    return response[0]["generated_text"].strip()



llm_output = generate_shap_explanation_detailed(
    prediction_score=prediction,
    is_defective=is_defective,
    shap_max=np.max(heatmap_raw),
    shap_mean=np.mean(heatmap_raw)
)

print("\n📝 LLM Explanation:\n", llm_output)

# === Show Only the SHAP Overlay Image ===

buf = BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
image_base64 = base64.b64encode(buf.read()).decode("utf-8")