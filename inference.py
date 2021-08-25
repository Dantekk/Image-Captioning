from utility import  get_inference_model, generate_caption
import json
import tensorflow as tf
import argparse
from settings_inference import *

# Get tokenizer layer from disk
tokenizer = tf.keras.models.load_model(tokernizer_path)
tokenizer = tokenizer.layers[1]

# Get model
model = get_inference_model(get_model_config_path)

# Load model weights
model.load_weights(get_model_weights_path)

# Generate new caption from input image
parser = argparse.ArgumentParser(description="Image Captioning")
parser.add_argument('--image', help="Path to image file.")
image_path = parser.parse_args().image

with open(get_model_config_path) as json_file:
    model_config = json.load(json_file)

text_caption = generate_caption(image_path, model, tokenizer, model_config["SEQ_LENGTH"])
print("PREDICT CAPTION : %s" %(text_caption))
