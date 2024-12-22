from PIL import Image, ImageOps
from flask import Flask, request
from flask_cors import CORS
import tensorflow as tf
import io
#
# Helper libraries
import numpy as np

app = Flask(__name__)
CORS(app)
#
class_names = ['T-shirt/top', 'Pantalon', 'Pullover', 'Robe', 'Manteau',
               'Sandale', 'Chemise', 'Sneaker', 'Sac', 'Bottine']


def loadAnnModel():
    model = tf.keras.models.load_model("./static/AnnModel.keras")
    return model


def loadCnnModel():
    model = tf.keras.models.load_model("static/cnnModel.keras")
    return model


def processAnnImage(img,width):
    size = width
    image = img.resize((size, size))
    image = image.convert('L')
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def processCnnImage(img,width):
    size = width
    image = img.resize((size, size))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predictAnnImage(img):
    model = loadAnnModel()
    img = processAnnImage(img,28)
    predictions = model.predict(img)
    top_class = class_names[np.argmax(predictions[0])]
    return f"Predicted class {top_class} with a Probability of {np.max(predictions):.2f} 🙂"


def predictCnnImage(img):
    model = loadCnnModel()
    img = processCnnImage(img,150)
    predictions = model.predict(img)
    print(predictions)

@app.route('/annModel', methods=["POST", "GET"])
def processAnnModel():
    if request.method == "POST":  # put application's code here
        try:
            img = request.files['imagefile']
            img_data = Image.open(io.BytesIO(img.read()))
            return_result = predictAnnImage(img_data)
            print(return_result)
            return return_result
        except Exception as e:
            print(e)
            return "Ann EndPoint has happened bad"
    else:
        return "WElcome To Imag"


@app.route("/cnnModel", methods=["POST", "GET"])
def processCnnModel():
    if request.method == "POST":
        try:
            img = request.files['imagefile']
            img_data = Image.open(io.BytesIO(img.read()))
            return_result = predictCnnImage(img_data)
            print(return_result)
            return return_result
        except Exception as e:
            print(e)
            return "Cnn EndPoint has happened bad"


@app.route("/", methods=["GET"])
def home():
    return "<h1>Hello You</h1>"


if __name__ == '__main__':
    app.run(debug=True,host="192.168.11.104")
