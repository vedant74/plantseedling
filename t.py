import keras
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
from flask import jsonify
from keras.models import load_model
import numpy as np

app = flask.Flask(__name__)
def create_model():

		model = keras.applications.ResNet50(input_shape=(256,256, 3), classes=2, weights=None)
		return model

def load_trained_model(weights_path):
		model = create_model()
		model.load_weights(weights_path)
		return model

def prepare_image(image):

	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize((256,256))
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	return image

def get(p):
	if p[0][1] >= 0.9996:
		return jsonify({'prediction': str(1)})
	else:

		return jsonify({'prediction': str(0)})

@app.route("/predict", methods=["POST"])
def predict():

	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image)

			# classify the input image and then initialize the list
			# of predictions to return to the client
			model=load_trained_model('C:/Users/vedan/Desktop/unschool_minor/top.h5')
			preds = model.predict(image)
			preds=np.array(preds)

	if preds[0][1] >= 0.9996:
		return jsonify({'prediction': str(1)})
	else:
			return jsonify({'prediction': str(0)})

if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	# model=load_trained_model('C:/Users/vedan/Desktop/unschool_minor/un.h5')
	app.run()
