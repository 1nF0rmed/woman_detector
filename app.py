from bottle import route, run, post, get, hook, response, request
import bottle
import warnings
from paste import httpserver
import sys
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import cv2
from keras.models import load_model
import numpy as np
import urllib, cStringIO
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input

warnings.filterwarnings('ignore')
detection_model_path = 'haarcascade_frontalface_default.xml'
gender_model_path = 'simple_CNN.81-0.96.hdf5'
gender_labels = get_labels('imdb')
# CONSTANTS
gender_offsets = (30, 60)
gender_offsets = (10, 10)
emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

# loading models
face_detection = load_detection_model(detection_model_path)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
gender_target_size = gender_classifier.input_shape[1:3]

@hook('after_request')
def enable_cors():
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Origin, Content-Type, Accept, X-Requested-With"

@route('/', method='OPTIONS')
@route('/<path:path>', method='OPTIONS')
def options_handler(path=None):
    return

# Handle image url
@post('/checkFemaleUrl')
def handleRequest():
    image_url = ""

    try:
        image_url = request.forms.get('image_url')
        print "THE IMAGE: ",image_url
    except:
        print "[ERR] Dead Image"
        return "-1"

    image_path = cStringIO.StringIO(urllib.urlopen(image_url).read())
    # loading images
    rgb_image = load_image(image_path, grayscale=False)
    gray_image = load_image(image_path, grayscale=True)
    gray_image = np.squeeze(gray_image)
    gray_image = gray_image.astype('uint8')

    faces = detect_faces(face_detection, gray_image)
    for face_coordinates in faces:
        print "Inside"
        x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]

        try:
            rgb_face = cv2.resize(rgb_face, (gender_target_size))
        except:
            continue

        rgb_face = preprocess_input(rgb_face, False)
        rgb_face = np.expand_dims(rgb_face, 0)
        gender_prediction = gender_classifier.predict(rgb_face)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]

        print "Predicted Gender: ", gender_text

        if gender_text == gender_labels[0]:
            print "[LOG] Man"

        else:
            print "[LOG] Woman"
            return "1"

    return "0"

# Space for other methods
run(host='0.0.0.0', port=8080, debug=True)
#application = bottle.default_app()
#httpserver.serve(application, host='0.0.0.0', port=8070)
