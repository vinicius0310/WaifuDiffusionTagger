# Waifu Diffusion Tagger

A web-based application for automatically tagging anime-style images using state-of-the-art deep learning models. This tool leverages various pre-trained models from the Waifu Diffusion project to identify content, characters, and other attributes in images.

## Features

- **Single Image Tagging**: Upload and analyze individual images through a user-friendly web interface
- **Batch Processing**: Process entire directories of images at once
- **Multiple Model Support**: Choose from various pre-trained models:
  - Dataset v3 models (newer):
    - SwinV2
    - ConvNext
    - ViT
    - ViT Large
    - EVA02 Large
  - Dataset v2 models:
    - MOAT
    - SwinV2
    - ConvNext
    - ConvNextV2
    - ViT
- **Customizable Thresholds**: Adjust confidence thresholds for general and character tags
- **Weighted Captions**: Option to weight tags by confidence scores
- **REST API**: Programmatic access through API endpoints

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/vinicius0310/WaifuDiffusionTagger.git
   cd WaifuDiffusionTagger
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. The application will automatically create a `model_cache` directory to store downloaded models.

## Usage

### Running the Application

Start the web server:
```
python app.py
```

Then open your browser and navigate to `http://localhost:5000`

### Web Interface

1. Select a model from the dropdown menu
2. Configure tagging parameters:
   - General tag threshold
   - Character tag threshold
   - Enable/disable minimum cut thresholds
   - Enable/disable weighted captions
3. Upload an image or specify a directory for batch processing
4. View the generated tags and confidence scores

### API Endpoints

#### Single Image Prediction
```
POST /api/predict
```
Parameters (form-data):
- `image`: Image file
- `model`: Model repository name (optional, defaults to SwinV2 v3)
- `general_thresh`: Threshold for general tags (optional, defaults to 0.35)
- `general_mcut_enabled`: Enable minimum cut for general tags (optional, defaults to false)
- `character_thresh`: Threshold for character tags (optional, defaults to 0.85)
- `character_mcut_enabled`: Enable minimum cut for character tags (optional, defaults to false)
- `weighted_captions`: Enable weighted captions (optional, defaults to false)

#### Batch Processing
```
POST /api/batch
```
Parameters (form-data):
- `directory`: Path to directory containing images
- `model`: Model repository name (optional, defaults to SwinV2 v3)
- `general_thresh`: Threshold for general tags (optional, defaults to 0.35)
- `general_mcut_enabled`: Enable minimum cut for general tags (optional, defaults to false)
- `character_thresh`: Threshold for character tags (optional, defaults to 0.85)
- `character_mcut_enabled`: Enable minimum cut for character tags (optional, defaults to false)
- `append_to_existing`: Append to existing tag files (optional, defaults to false)
- `exclude_character_tags`: Exclude character tags (optional, defaults to false)
- `weighted_captions`: Enable weighted captions (optional, defaults to false)

## Configuration

The application uses the following default configuration:
- Upload folder: `uploads/`
- Maximum upload size: 16MB
- Model cache directory: `model_cache/`
- Default model: SwinV2 v3
- Default general tag threshold: 0.35
- Default character tag threshold: 0.85

## Requirements

See `requirements.txt` for a complete list of dependencies. Main requirements include:
- Flask
- Pillow (PIL)
- NumPy
- Transformers (for model loading)
- PyTorch (for inference)

## Credits

This application uses models developed by SmilingWolf for the Waifu Diffusion project. Visit the model repositories for more information:
- [SmilingWolf's Hugging Face profile](https://huggingface.co/SmilingWolf)
