from flask import Flask, render_template, request
from utils import process_image
import os

app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return "No file uploaded!", 400

    file = request.files["image"]
    if file.filename == "":
        return "No file selected!", 400

    # Save the file to the upload folder
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Process the image
    class_name, confidence_score = process_image(file_path)

    return render_template(
        "result.html",
        class_name=class_name,
        confidence_score=confidence_score,
        image_path=file_path,
    )

if __name__ == "__main__":
    app.run(debug=True)
