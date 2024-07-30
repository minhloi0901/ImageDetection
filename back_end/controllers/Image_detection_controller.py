# Image_detection_controller.py
from flask import jsonify, request
from models.Image_detection_model import ImageDetectionModel

def detect_image():
    # Get image from request
    image_file = request.files('image')
    # test get image from D:\Coding\anngu\ImageDetection\phusi.png
    image_file = open('D:\Coding/anngu\ImageDetection\phusi.png', 'rb')
    
    if image_file:
        return ImageDetectionModel.detect_image(image_file)
    else:
        return jsonify({
            'error': 'Image not found'
        }), 400
    