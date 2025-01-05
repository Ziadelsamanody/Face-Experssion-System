# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
model = load_model('modelaugv1.h5')

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize to 48x48 pixels
    img = cv2.resize(img, (48, 48))
    # Normalize pixel values
    img = img / 255.0
    # Reshape for model input
    img = np.reshape(img, (1, 48, 48, 1))
    return img

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image and make prediction
        processed_img = preprocess_image(filepath)
        predictions = model.predict(processed_img)[0]
        
        # Get top emotion and probability
        top_emotion = EMOTIONS[np.argmax(predictions)]
        probability = float(np.max(predictions))
        
        # Get all emotions and their probabilities
        all_emotions = {emotion: float(pred) for emotion, pred in zip(EMOTIONS, predictions)}
        
        return jsonify({
            'success': True,
            'filename': filename,
            'top_emotion': top_emotion,
            'probability': probability,
            'all_emotions': all_emotions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run() 