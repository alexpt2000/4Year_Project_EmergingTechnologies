from flask import Flask, render_template, request
import numpy as np
import keras.models
import re
import base64
from scipy.misc import imread, imresize
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import sys 
import os

app = Flask(__name__)
global model, graph

def init():
    num_classes = 10
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # load woeights into new model
    model.load_weights("weights.h5")


    # compile and evaluate loaded model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    #loss, accuracy = model.evaluate(X_test,y_test)
    #print('loss:', loss)
    # print('accuracy:', accuracy)
    graph = tf.get_default_graph()

    return model, graph



model, graph = init()
    
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/ImageResult/', methods=['GET','POST'])
def predict():

    parseImage(request.get_data())


    x = imread('static/ImageResult.png', mode='L')
    x = np.invert(x)
    x = imresize(x,(28,28))


    x = x.reshape(1,28,28,1)
    with graph.as_default():
        out = model.predict(x)
        #print(out)
        print(np.argmax(out, axis=1))
        response = np.array_str(np.argmax(out, axis=1))
        return response 
    
def parseImage(imgData):
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('static/ImageResult.png','wb') as output:
        output.write(base64.decodebytes(imgstr))


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 1111))
    app.run(host='0.0.0.0', port=port)