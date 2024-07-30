# Image_detection_controller.py
from flask import jsonify, request
from models.Image_detection_model import ImageDetectionModel
from PIL import Image

def detect_image():
    # Get image from request
    image_file = request.files.get('image')
    # test get image from D:\Coding\anngu\ImageDetection\phusi.png
    image_file = open('D:\Coding/anngu\ImageDetection\phusi.png', 'rb')
    # show image
    image = Image.open(image_file)
    image.show()

    if image_file:
        return ImageDetectionModel.detect_image(image_file)
    else:
        return jsonify({
            'error': 'Image not found'
        }), 400
    