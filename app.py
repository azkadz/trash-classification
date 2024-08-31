from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import io
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained model
model = load_model('model.h5')

def prepare_image(image):
    img = image.resize((128, 128))
    img_array = np.array(img).reshape((1, 128, 128, 3))
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    img_array = prepare_image(image)
    
    prediction = model.predict(img_array)
    label = 'organic' if prediction > 0.5 else 'inorganic'
    
    return jsonify({'classification': label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
