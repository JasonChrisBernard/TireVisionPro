
# ✅ Import Required Libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf 
import numpy as np
import cv2
import os
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from lime import lime_image
from skimage.segmentation import slic, mark_boundaries
from transformers import pipeline
import shap
import random




# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


MODEL_PATH = "/mnt/d/IIT/Degree/4th year/Final Year Project/Tire detection System/TireVisionPro/InceptionResNetV2_tire_defect_model_TPU_2.keras"

print(tf.__version__)

if os.path.exists(MODEL_PATH):
    print("✅ File exists!")
else:
    print("❌ File NOT found!")


try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully! Type:", type(model))
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model! Error: {e}")



# ✅ Load LLM (Facebook BART for Summarization)
generator = pipeline("text2text-generation", model="facebook/bart-large-cnn", device= -1)


@app.route('/')
def home():
    return "✅ TireVisionPro Backend is Running! Use /predict to make predictions."

# ✅ Image Preprocessing Function
def preprocess_image(img):
    img = cv2.resize(img, (299, 299))  # Resize to model input size
    img_array = np.expand_dims(img / 255.0, axis=0)  # Normalize
    return img_array

# ✅ Function to Generate Grad-CAM Heatmap
def generate_gradcam(model, img_array):
    last_conv_layer_name = "conv_7b"
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # Assuming binary classification

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    for i in range(conv_outputs.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv2.resize(heatmap, (299, 299))

    return heatmap

# ✅ Function to Overlay Heatmap on Image
def overlay_heatmap(img, heatmap):
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return overlay_img

# ✅ Function to Apply LIME and return superpixel insights
def apply_lime(model, img_array):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array[0],
        classifier_fn=lambda x: model.predict(x),
        top_labels=1,
        hide_color=0,
        num_samples=1000,
        segmentation_fn=lambda x: slic(x, n_segments=100, compactness=10)
    )

    label = explanation.top_labels[0]  # Top predicted class
    _, mask = explanation.get_image_and_mask(label=label, positive_only=True, num_features=5, hide_rest=False)
    lime_img = mark_boundaries(img_array[0], mask)
    lime_img = (lime_img * 255).astype(np.uint8)

    lime_focus_count = np.sum(mask > 0)  # Number of pixels in important segments
    lime_total_segments = len(np.unique(mask))  # Total number of segments from SLIC

    return lime_img, lime_focus_count, lime_total_segments


def apply_shap_visual(model, img_array, is_defective):
    base_dir = "Defective" if is_defective else "Good"
    bg_dir = f"/mnt/d/IIT/Degree/4th year/Final Year Project/Tire detection System/TireVisionPro/Tyre Classification Gray Scale/train/{base_dir}"

    all_files = [f for f in os.listdir(bg_dir) if f.lower().endswith((".jpg", ".png"))]
    random.shuffle(all_files)

    background_images = []
    for fname in all_files[:100]:
        path = os.path.join(bg_dir, fname)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (299, 299)) / 255.0
            background_images.append(img)

    background_array = np.array(background_images)

    with tf.device("/CPU:0"):
        explainer = shap.GradientExplainer((model.input, model.output), background_array)
        shap_values = explainer.shap_values(img_array)

    shap_array = shap_values[0][0]
    heatmap_raw = np.sum(np.abs(shap_array), axis=-1)
    heatmap_norm = np.uint8(255 * heatmap_raw / (np.max(heatmap_raw) + 1e-8))
    heatmap_blurred = cv2.GaussianBlur(heatmap_norm, (5, 5), 0)
    heatmap_colored = cv2.applyColorMap(heatmap_blurred, cv2.COLORMAP_JET)

    # 🔧 Ensure shape & channel compatibility
    original_img = (img_array[0] * 255).astype(np.uint8)
    if original_img.shape[:2] != heatmap_colored.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, (original_img.shape[1], original_img.shape[0]))

    if len(original_img.shape) != 3 or original_img.shape[2] != 3:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

    if len(heatmap_colored.shape) != 3 or heatmap_colored.shape[2] != 3:
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_GRAY2BGR)

    # ✅ Now safe to overlay
    overlay = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
    shap_max = np.max(heatmap_raw)
    shap_mean = np.mean(heatmap_raw)

    return overlay, shap_max, shap_mean  # ✅ Return all 3 values!



# ✅ Function to Generate LLM Explanation
def generate_GradCam_explanation(prediction_score, gradcam_focus, is_defective):
    explanation_prompt = f"""
    The AI tire detection system analyzed an image and provided the following results:

    - **Prediction Score:** {prediction_score:.4f}
    - **Grad-CAM Focus Area:** {gradcam_focus}
    - **Defective Condition:** {'Yes' if is_defective else 'No'}

    ### **Analysis**
    1️⃣ **Detection Results:** What patterns did the AI see?
    2️⃣ **Grad-CAM Focus:** How does the heatmap confirm this?
    3️⃣ **Condition Breakdown:** What are the defects (if any)?
    4️⃣ **Final Conclusion:** Why is the tire {'defective' if is_defective else 'good'}?
    5️⃣ **Actionable Steps:** What should the user do next?
    """

    if is_defective:
        recommendations = "- 🚨 **Replace the tire immediately!** Severe tread wear or structural issues detected.\n" \
                          "- 🔍 **Manual inspection required** for hidden damages.\n" \
                          "- ✅ **Ensure proper tire pressure & wheel alignment** to prevent future damage."
    else:
        recommendations = "- ✅ **Tire is in good condition.** No major wear detected.\n" \
                          "- 🔍 **Regular maintenance recommended** for longevity.\n" \
                          "- 🛠️ **Check air pressure monthly** to ensure safety."

    full_prompt = explanation_prompt + "\n### **✅ Recommended Actions:**\n" + recommendations

    response = generator(full_prompt, max_new_tokens=1000, num_return_sequences=1, truncation=True, temperature=0.8)
    return response[0]["generated_text"].strip()

# ✅ Function to Generate LIME LLM Explanation
def generate_lime_explanation(prediction_score, is_defective, lime_focus_count, lime_total_segments):
    explanation_prompt = f"""
    The AI tire defect detection system used **LIME (Local Interpretable Model-Agnostic Explanations)** to understand its prediction.

    - **Prediction Score:** {prediction_score:.4f}
    - **Defective Condition:** {"Yes" if is_defective else "No"}
    - **Highlighted Superpixels:** {lime_focus_count} of {lime_total_segments} total segments

    ### 🧠 LIME-Based Interpretation
    1️⃣ **Influential Areas:** LIME marked {lime_focus_count} regions as key to the prediction.
    2️⃣ **Explanation Clarity:** These areas are likely near the tire's surface patterns like treads or bulges.
    3️⃣ **Interpretation Support:** This supports the model's confidence level.
    4️⃣ **Final Verdict:** The tire is predicted as {'defective' if is_defective else 'good'} based on these focus zones.
    """

    if is_defective:
        recommendations = "- 🔴 **Replace the tire immediately.** Critical superpixels indicate damage.\n" \
                          "- 🛠️ **Visually inspect highlighted zones for cracks or wear.**\n" \
                          "- 🚫 **Avoid long drives with this tire.**"
    else:
        recommendations = "- ✅ **No defect zones found.** LIME supports a safe classification.\n" \
                          "- 🔍 **Keep inspecting the tire routinely.**\n" \
                          "- 🧰 **LIME confirms healthy surface patterns.**"

    full_prompt = explanation_prompt + "\n### ✅ Recommended Actions:\n" + recommendations

    response = generator(full_prompt, max_new_tokens=1000, num_return_sequences=1, truncation=True, temperature=0.8)
    return response[0]["generated_text"].strip()

def generate_shap_explanation(prediction_score, shap_max, shap_mean, is_defective):
    explanation_prompt = f"""
    The AI tire detection system used SHAP (SHapley Additive exPlanations) to interpret the model’s decision.

    - **Prediction Score:** {prediction_score:.4f}
    - **Max SHAP Value:** {shap_max:.4f}
    - **Mean SHAP Value:** {shap_mean:.4f}
    - **Defective Condition:** {'Yes' if is_defective else 'No'}

    ### **Analysis**
    1️⃣ **SHAP Insights:** Which regions influenced the decision?
    2️⃣ **High Contribution Zones:** What do the SHAP values indicate?
    3️⃣ **Interpretation Meaning:** How does SHAP support the model's prediction?
    4️⃣ **Final Conclusion:** Why is the tire classified as {'defective' if is_defective else 'good'}?
    5️⃣ **Actionable Steps:** What should the user do next?
    """

    if is_defective:
        recommendations = "- 🚨 **Replace the tire immediately.** High SHAP values suggest visible or hidden damage.\n" \
                          "- 🧐 **Inspect high-impact areas** indicated in the SHAP overlay.\n" \
                          "- ✅ **Consider routine checks** to prevent future failures."
    else:
        recommendations = "- ✅ **Tire appears healthy.** SHAP confirms the model’s safe decision.\n" \
                          "- 🔍 **Continue regular inspections** for long-term safety.\n" \
                          "- 🛞 **Monitor tire pressure and tread condition** monthly."

    full_prompt = explanation_prompt + "\n### **✅ Recommended Actions:**\n" + recommendations

    response = generator(full_prompt, max_new_tokens=1000, num_return_sequences=1, truncation=True, temperature=0.8)
    return response[0]["generated_text"].strip()


# ✅ API Endpoint for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Check if File Exists in Request
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # ✅ Fix: Check if Image Was Read Correctly
        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        # ✅ Preprocess Image
        img_array = preprocess_image(img)

        print("🔍 Model is:", model)
        print("✅ img_array shape:", img_array.shape)

        if model is None:
            raise RuntimeError("❌ Model is None at prediction time!")

        # ✅ Model Prediction
        prediction = model.predict(img_array)[0][0]
        is_defective = prediction < 0.3  # Adjust threshold
        defect_status = "🚨 Defective Tire Detected! ⚠️" if is_defective else "✅ Tire is in Good Condition!"
        gradcam_focus = "Tire Treads"

        # ✅ Generate Grad-CAM
        heatmap = generate_gradcam(model, img_array)
        gradcam_img = overlay_heatmap(cv2.resize(img, (299, 299)), heatmap)

       
        # ✅ Generate LIME
        lime_img, lime_focus_count, lime_total_segments = apply_lime(model, img_array)


        # ✅ Convert Images to Base64
        _, buffer = cv2.imencode(".jpg", gradcam_img)
        gradcam_base64 = base64.b64encode(buffer).decode("utf-8")

        _, buffer = cv2.imencode(".jpg", lime_img)
        lime_base64 = base64.b64encode(buffer).decode("utf-8")

        # ✅ Generate LLM Explanation
        explanation = generate_GradCam_explanation(prediction, gradcam_focus, is_defective)

        lime_explanation = generate_lime_explanation(prediction, is_defective, lime_focus_count, lime_total_segments)
        print("🔍 LIME Focused Segments:", lime_focus_count)
        print("🧠 LIME Explanation:", lime_explanation)

        # ✅ Apply SHAP and Convert to Base64
        shap_overlay ,shap_max, shap_mean= apply_shap_visual(model, img_array, is_defective)
        _, buffer = cv2.imencode(".jpg", shap_overlay)
        shap_base64 = base64.b64encode(buffer).decode("utf-8")

        shap_explanation = generate_shap_explanation(
            prediction_score=prediction,
            shap_max=shap_max,
            shap_mean=shap_mean,
            is_defective=is_defective
        )


        return jsonify({
            "defect_status": defect_status,
            "probability": float(prediction),
            "gradcam": gradcam_base64,
            "lime": lime_base64,
            "shap": shap_base64,  # 🆕 SHAP image only
            "explanation": explanation,              # Grad-CAM LLM explanation
            "lime_explanation": lime_explanation,     # ✅ Add this
            "shap_explanation": shap_explanation

        })


    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ Run Flask App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)