from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import base64
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load the Keras model
model = load_model("waste_classification_model_optimized.h5")

# Class labels and bin images
class_labels = ["Organic Waste", "Recyclable Waste"]
bin_images = {
    "Organic Waste": "static/029670cc42af7e14d1ebb6e966a9e0cd.png",
    "Recyclable Waste": "static/noun-organic-waste-4011863-007435.png"
}

# Global counters for waste classification
organic_count = 0
recyclable_count = 0

# Prepare the image for prediction
def prepare_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array.astype("float32") / 255.0
    return image_array

# Classify the image
def classify_image(image):
    global organic_count, recyclable_count
    img = prepare_image(image)
    pred = model.predict(img)
    label = "Organic Waste" if pred[0][0] < 0.5 else "Recyclable Waste"
    confidence = pred[0][0] if pred[0][0] < 0.5 else 1 - pred[0][0]

    # Update the counts based on the classification
    if label == "Organic Waste":
        organic_count += 1
    elif label == "Recyclable Waste":
        recyclable_count += 1

    return label, confidence

# Convert an image to Base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    # Determine the source of the image
    if "file" in request.files:
        file = request.files["file"]
        image = Image.open(file.stream)
    elif "image_base64" in request.json:
        image_data = request.json["image_base64"]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    else:
        return jsonify({"error": "No image provided"}), 400

    # Classify the image
    label, confidence = classify_image(image)
    bin_image_url = bin_images.get(label, "")

    # Convert the uploaded/captured image to Base64 for preview
    image_base64 = image_to_base64(image)

    return jsonify({
        "label": label,
        "confidence": float(confidence),  # Convert to native Python float
        "bin_image_url": bin_image_url,
        "preview_image_base64": f"data:image/png;base64,{image_base64}"
    })

@app.route("/get_counts", methods=["GET"])
def get_counts():
    # Return the current counts of organic and recyclable waste
    return jsonify({
        "organic_count": organic_count,
        "recyclable_count": recyclable_count
    })

if __name__ == "__main__":
    app.run(debug=True)
