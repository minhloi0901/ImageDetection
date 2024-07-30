# Image_detection_route.py
from flask import Blueprint, request, jsonify
from controllers import Image_detection_controller

image_detection_router = Blueprint('image_detection', __name__)

@image_detection_router.route('/', methods=['GET'])
def root():
    return 'Image detection service'

@image_detection_router.route('/image', methods=['GET'])
def detect_image():
    return Image_detection_controller.detect_image()
