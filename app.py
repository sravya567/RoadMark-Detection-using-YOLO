from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import os
from ultralytics import YOLO
app = Flask(__name__)
model = YOLO("best.pt")
@app.route('/')
def index():
    return render_template('index.html')
@app.route("/inspect")
def inspect():
    return render_template('inner-page.html')
@app.route('/upload', methods=["POST"])
def detect():
    if request.method == "POST":
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'Uploads', f.filename)
        f.save(filepath)
        im1 = Image.open(filepath)
        results = model.predict(source=im1, save=True)
        print(results)
        return display()
@app.route('/display')
def display():
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    if subfolders:
        latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
        directory = os.path.join(folder_path, latest_subfolder)
        files = os.listdir(directory)
        if files:
            latest_file = files[0]
            filename = os.path.join(folder_path, latest_subfolder, latest_file)
            file_extension = filename.rsplit('.', 1)[1].lower()
            if file_extension in {'jpg'}:
                return send_from_directory(directory, latest_file)
    return "No valid files found."
if __name__ == '__main__':
    app.run(debug=True)