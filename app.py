import time

# get flask stuff
from flask import Flask, render_template, request, jsonify

# get maths
import numpy as np
import tensorflow as tf

app = Flask(__name__)



@app.route("/")
def index():
    return render_template("home.html")

def do_predict(data):
    # get the model
    loadS = time.time()
    model = tf.keras.models.load_model("test_model.h5")
    loadE = time.time()
    
    computeS = time.time()
    # put data into numpy
    img = np.array(data)
    # need to normalize it
    img = img / 255.0
    # also need to tensorify it
    img = img.reshape((1,28,28))
    res = np.argmax(model.predict(img))
    computeE = time.time()
    return (res, loadE-loadS, computeE-computeS)

@app.route("/recog", methods=["POST"])
def recog():
    img = request.json.get("data")
    res, loadT, computeT = do_predict(img)
    return jsonify({"answer": int(res), "loadT": loadT, "computeT": computeT})

if __name__ == "__main__":
    app.run(port=5000, debug=True)