import os
import json
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
from predictor import Predictor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Constants for model configuration
MODEL_CACHE_DIR = "./model_cache"

# Dataset v3 series of models:
SWINV2_MODEL_DSV3_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
CONV_MODEL_DSV3_REPO = "SmilingWolf/wd-convnext-tagger-v3"
VIT_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-tagger-v3"
VIT_LARGE_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-large-tagger-v3"
EVA02_LARGE_MODEL_DSV3_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"

# Dataset v2 series of models:
MOAT_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
SWIN_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
CONV_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
CONV2_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
VIT_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"

# Initialize predictor
predictor = Predictor(MODEL_CACHE_DIR)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    models = [
        SWINV2_MODEL_DSV3_REPO,
        CONV_MODEL_DSV3_REPO,
        VIT_MODEL_DSV3_REPO,
        VIT_LARGE_MODEL_DSV3_REPO,
        EVA02_LARGE_MODEL_DSV3_REPO,
        MOAT_MODEL_DSV2_REPO,
        SWIN_MODEL_DSV2_REPO,
        CONV_MODEL_DSV2_REPO,
        CONV2_MODEL_DSV2_REPO,
        VIT_MODEL_DSV2_REPO,
    ]
    return render_template('index.html', models=models)

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    # Get parameters from form
    model_repo = request.form.get('model', SWINV2_MODEL_DSV3_REPO)
    general_thresh = float(request.form.get('general_thresh', 0.35))
    general_mcut_enabled = request.form.get('general_mcut_enabled', 'false').lower() == 'true'
    character_thresh = float(request.form.get('character_thresh', 0.85))
    character_mcut_enabled = request.form.get('character_mcut_enabled', 'false').lower() == 'true'
    weighted_captions = request.form.get('weighted_captions', 'false').lower() == 'true'
    
    # Process the image
    try:
        image = Image.open(file_path)
        tags, rating, character_res, general_res = predictor.predict(
            image,
            model_repo,
            general_thresh,
            general_mcut_enabled,
            character_thresh,
            character_mcut_enabled,
            weighted_captions,
        )
        
        # Convert numpy values to Python types for JSON serialization
        rating = {k: float(v) for k, v in rating.items()}
        character_res = {k: float(v) for k, v in character_res.items()}
        general_res = {k: float(v) for k, v in general_res.items()}
        
        return jsonify({
            'tags': tags,
            'rating': rating,
            'character_tags': character_res,
            'general_tags': general_res
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/api/batch', methods=['POST'])
def batch_process():
    if 'directory' not in request.form:
        return jsonify({'error': 'No directory provided'}), 400
    
    image_dir = request.form.get('directory')
    if not os.path.isdir(image_dir):
        return jsonify({'error': 'Invalid directory path'}), 400
    
    # Get parameters from form
    model_repo = request.form.get('model', SWINV2_MODEL_DSV3_REPO)
    general_thresh = float(request.form.get('general_thresh', 0.35))
    general_mcut_enabled = request.form.get('general_mcut_enabled', 'false').lower() == 'true'
    character_thresh = float(request.form.get('character_thresh', 0.85))
    character_mcut_enabled = request.form.get('character_mcut_enabled', 'false').lower() == 'true'
    append_to_existing = request.form.get('append_to_existing', 'false').lower() == 'true'
    exclude_character_tags = request.form.get('exclude_character_tags', 'false').lower() == 'true'
    weighted_captions = request.form.get('weighted_captions', 'false').lower() == 'true'
    
    # Process the batch
    try:
        results = predictor.batch_predict(
            image_dir,
            model_repo,
            general_thresh,
            general_mcut_enabled,
            character_thresh,
            character_mcut_enabled,
            append_to_existing,
            exclude_character_tags,
            weighted_captions,
        )
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)