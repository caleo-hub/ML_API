from flask import Flask, request, jsonify
from app.torch_utils import transform_image, get_prediction
app = Flask(__name__)


def get_class_name(prediction):
    classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"]
    
    return classes[prediction]

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file_input = request.files.get('file')
        
        if file_input is None or file_input.filename == "":
            return jsonify({'error': 'No file'})
    
    try:
        image_bytes = file_input.read()
        image_tensor = transform_image(image_bytes)
        prediction = get_prediction(image_tensor)
        data = {'prediction': prediction.item(), 'class_name': get_class_name(prediction.item())}
        return jsonify(data)
    except Exception as e:
        return jsonify(({'error': 'internal error'}))
    
