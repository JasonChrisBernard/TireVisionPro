import cv2
import numpy as np
import tensorflow as tf
import os

# 🛞 Hardcoded sample image path (✅ CHANGE this to your actual path)
sample_image_path = "/mnt/d/IIT/Degree/4th year/Final Year Project/Tire detection System/TireVisionPro/Tyre Classification Gray Scale/train/Good/good-668-_jpg.rf.23011e261d8c60deebe0daad53c68c97.jpg"

# ✅ Check if file exists
if not os.path.isfile(sample_image_path):
    raise ValueError("❌ File not found or path is invalid!")

# ✅ Load the model
MODEL_PATH = "/mnt/d/IIT/Degree/4th year/Final Year Project/Tire detection System/TireVisionPro/InceptionResNetV2_tire_defect_model_TPU_2.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Load and preprocess the selected image
img = cv2.imread(sample_image_path)
if img is None:
    raise ValueError("❌ Failed to load image.")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (299, 299))
img_array = np.expand_dims(img / 255.0, axis=0)

# ✅ Run prediction
prediction = model.predict(img_array)[0][0]
print("\n✅ Prediction Score:", prediction)

# ✅ Interpret result
if prediction < 0.3:
    print("🚨 Defective Tire Detected! ⚠️")
else:
    print("✅ Tire is in Good Condition!")
