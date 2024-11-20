from flask import Flask, request, jsonify
from flask_cors import CORS  # For enabling CORS
import os
import base64
import requests
import logging
from datetime import datetime
from io import BytesIO
from PIL import Image

# Constants
LOG_DIR = "logs"
LOG_FILE_PATH = os.path.join(LOG_DIR, "api_requests.log")

# Ensure necessary directories exist
os.makedirs(LOG_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)

# Flask app setup
app = Flask(__name__)
CORS(app)  # Enable CORS

# Function to convert an image file to base64
def image_file_to_base64(image_file):
    with Image.open(image_file) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Function to log API responses to a file
def log_response(status_code, response_content):
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write(f"{datetime.now()} - Status Code: {status_code}\nResponse: {response_content}\n\n")

# Function to send the API request to Try-On Diffusion
def send_tryon_request(model_image_file, cloth_image_file, api_key):
    model_image_base64 = image_file_to_base64(model_image_file)
    cloth_image_base64 = image_file_to_base64(cloth_image_file)

    data = {
        "model_image": model_image_base64,
        "cloth_image": cloth_image_base64,
        "category": "Upper body",  # Example category
        "num_inference_steps": 35,
        "guidance_scale": 2,
        "seed": 12467,
        "base64": True
    }

    headers = {'x-api-key': api_key}

    try:
        # API Request to the Try-On Diffusion service
        response = requests.post("https://api.segmind.com/v1/try-on-diffusion", json=data, headers=headers)
        log_response(response.status_code, response.content)

        if response.status_code == 200:
            response_json = response.json()
            if 'image' in response_json:
                image_base64 = response_json['image']
                return {"status": "success", "output_image": image_base64}
            else:
                return {"status": "error", "message": "No image data found in response."}
        else:
            return {"status": "error", "message": f"API request failed with status code {response.status_code}"}
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return {"status": "error", "message": str(e)}

# Flask route to handle the try-on image generation
@app.route('/generate-tryon', methods=['POST'])
def generate_tryon():
    model_image = request.files.get('model_image')
    cloth_image = request.files.get('cloth_image')
    api_key = os.getenv("API_KEY", "your-api-key-here")  # Get the API key from environment variable or hardcode here if you prefer
    
    if not model_image or not cloth_image:
        return jsonify({"status": "error", "message": "Model or Cloth image not found."}), 400
    
    # Send the images directly to the Try-On Diffusion API
    result = send_tryon_request(model_image, cloth_image, api_key)
    
    if result["status"] == "success":
        # Return the generated image as a base64 string
        return jsonify(result)
    else:
        # Return error if the generation failed
        return jsonify(result), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
