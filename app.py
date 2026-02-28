import io
from flask import Flask, request, jsonify
from flask_cors import CORS 
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('best.pt')

@app.route('/api/scan', methods=['POST'])
def scan_inventory():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    results = model(img)
    counts = {}
    class_names = results[0].names 

    for class_id in results[0].boxes.cls:
        class_name = class_names[int(class_id)]
        counts[class_name] = counts.get(class_name, 0) + 1

    print("Detected counts:", counts)
    return jsonify(counts)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)