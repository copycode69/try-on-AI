import os
import base64
import requests
import logging
from datetime import datetime
from pathlib import Path

# Constants for directories and file paths
INPUT_DIR = "images/input"
OUTPUT_DIR = "images/output"
LOG_DIR = "logs"
OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, "swapped_output.png")
LOG_FILE_PATH = os.path.join(LOG_DIR, "api_requests.log")

# Ensure necessary directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)

# Retrieve API Key from environment variable
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API key not found. Set the API key in your environment variables.")

# Function to convert an image file to base64
def image_file_to_base64(image_path):
    """Convert an image file to base64 encoding."""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')

# Function to log API responses to a file
def log_response(status_code, response_content):
    """Log the status code and response content into the log file."""
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write(f"{datetime.now()} - Status Code: {status_code}\nResponse: {response_content}\n\n")

# Function to check if input images exist
def check_images_exist(model_image_path, cloth_image_path):
    """Check if the input images exist at the specified paths."""
    if not os.path.exists(model_image_path):
        raise FileNotFoundError(f"Model image not found at {model_image_path}")
    if not os.path.exists(cloth_image_path):
        raise FileNotFoundError(f"Cloth image not found at {cloth_image_path}")

# Function to send the API request to Try-On Diffusion
def send_tryon_request(model_image_path, cloth_image_path, output_image_path):
    """Send the try-on request to the Try-On Diffusion API."""
    model_image_base64 = image_file_to_base64(model_image_path)
    cloth_image_base64 = image_file_to_base64(cloth_image_path)

    # Request payload
    data = {
        "model_image": model_image_base64,
        "cloth_image": cloth_image_base64,
        "category": "Upper body",  # Can change depending on clothing category
        "num_inference_steps": 35,
        "guidance_scale": 2,
        "seed": 12467,
        "base64": True
    }

    headers = {'x-api-key': API_KEY}

    try:
        # Sending the POST request to the API
        response = requests.post("https://api.segmind.com/v1/try-on-diffusion", json=data, headers=headers)
        log_response(response.status_code, response.content)

        if response.status_code == 200:
            try:
                # Parse JSON response
                response_json = response.json()
                if 'image' in response_json:
                    # Decode and save the generated image
                    image_base64 = response_json['image']
                    with open(output_image_path, "wb") as out_file:
                        out_file.write(base64.b64decode(image_base64))
                    print(f"Image saved successfully to {output_image_path}")
                else:
                    print("No image data found in response.")
            except ValueError as e:
                print("Error parsing JSON:", e)
        else:
            print(f"API request failed with status code: {response.status_code}")
            print(response.content)
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        print(f"Error: {e}")

# Main execution
if __name__ == "__main__":
    model_image_path = os.path.join(INPUT_DIR, "shortswithmen.jpg")
    cloth_image_path = os.path.join(INPUT_DIR, "shirt1.jpg")

    # Check that both input images exist
    try:
        check_images_exist(model_image_path, cloth_image_path)

        # Send request and process images
        send_tryon_request(model_image_path, cloth_image_path, OUTPUT_IMAGE_PATH)
    except FileNotFoundError as e:
        logging.error(str(e))
        print(str(e))
