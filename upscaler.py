# Importing basic libraries

import os
import time 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
os.environ['TFHUB_DOWNLOAD_PROGRESS']='True'

IMAGE_ORIGINAL_PATH = './pics_original'
IMAGE_UPSCALED_PATH = './pics_edited'
SAVED_MODEL_PATH = "./esrgan"

new_dpi = (300, 300)

# function to preprocess image so that it can be handled by model
def preprocess_image(image_path):
	'''Loads the image given make it ready for 
		the model
		Args:
		image_path: Path to the image file
	'''
	image = tf.image.decode_image(tf.io.read_file(image_path))
	if image.shape[-1] == 4:
		image = image[...,:-1]
	size = (tf.convert_to_tensor(image.shape[:-1]) // 4) * 4
	image = tf.image.crop_to_bounding_box(image, 0, 0, size[0], size[1])
	image = tf.cast(image,tf.float32)
	return tf.expand_dims(image,0)

def save_image(image,filename):
	''' 
	 Saves unscaled Tensor Images
	 image: 3D image Tensor
	 filename: Name of the file to be saved
	'''
	if not isinstance(image, Image.Image):
		image = tf.clip_by_value(image, 0, 255)
		image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
	image.save('%s' % filename, dpi=new_dpi)
	print('Saved as %s' % filename)


model = tf.saved_model.load(SAVED_MODEL_PATH)

# Start Performing resolution 
start = time.time()

for filename in os.listdir(IMAGE_ORIGINAL_PATH):
	if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
		file_path = os.path.join(IMAGE_ORIGINAL_PATH, filename)
		load_image = preprocess_image(file_path)
		super_image = model(load_image)
		super_image = tf.squeeze(super_image)
		save_image(super_image, os.path.join(IMAGE_UPSCALED_PATH, filename))
print('Time taken to complete process: %f'%(time.time() - start))
