from flask import Flask
from routers.Image_detection_route import image_detection_router

app = Flask(__name__)

app.register_blueprint(image_detection_router, url_prefix='/image_detection')

@app.route('/')
def root():
    return 'Server is running'

if __name__ == '__main__':
    app.run(debug=True) 