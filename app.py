import base64
import numpy as np
from matplotlib import pyplot
import tensorflow as tf
import sys
import os

from flask import request
from flask import Flask, render_template
from flask import jsonify
from flask_cors import CORS, cross_origin
from datetime import datetime 
import cv2
import imageio
from skimage.color import rgb2gray
from skimage.color import gray2rgb


global model_mantra, model_buster, graph


app = Flask(__name__)
cors = CORS(app, resources={r"/foo": {"origins": "http://localhost:port"}})

#Load ManTraNet
manTraNet_root = 'ManTraNet/'
manTraNet_srcDir = os.path.join( manTraNet_root, 'src' )
sys.path.insert( 0, manTraNet_srcDir )
manTraNet_modelDir = os.path.join( manTraNet_root, 'pretrained_weights' )

from ManTraNet.src import modelCore
model_mantra = modelCore.load_pretrain_model_by_index( 4, manTraNet_modelDir )

#Load BusterNet
busterNet_root = 'BusterNet/'
busterNet_srcDir = os.path.join( busterNet_root, 'Model' )
sys.path.insert( 0, busterNet_srcDir )

from BusterNet.Model.BusterNetCore import create_BusterNet_testing_model
model_buster = create_BusterNet_testing_model( 'BusterNet/Model/pretrained_busterNet.hd5' )
from BusterNet.Model.BusterNetUtils import *

#Set FLASK Thread
graph = tf.get_default_graph() 

@app.route('/', methods=['GET', 'POST'])
def home():
    print("here")
    return render_template('predict.html')


@app.route('/predict', methods=['GET', 'POST'])
@cross_origin(origin='localhost', headers=['Content- Type', 'Authorization'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image'].split(',')[1]
    print("-----------Received image--------")
    
    with open("imageToSave.png", "wb") as fh:
        fh.write(base64.b64decode(encoded))
        
    img = cv2.imread('imageToSave.png')
    img = cv2.resize(img, dsize=(640,480), interpolation=cv2.INTER_CUBIC)
    buster_mask, btime = buster_pred(img, model_buster)
    buster_mask = rgb2gray(buster_mask)
    pyplot.figure( figsize=(15,5) )
    pyplot.subplot(131)
    pyplot.imshow( img )
    pyplot.subplot(132)
    pyplot.imshow( buster_mask, cmap='gray' )
    pyplot.show()
    
    #ManTraNet
    rgb, mask, ptime = decode_an_image_file( 'imageToSave.png', model_mantra )
    pyplot.figure( figsize=(15,5) )
    pyplot.subplot(131)
    pyplot.imshow( rgb )
    pyplot.subplot(132)
    pyplot.imshow( mask, cmap='gray' )
    pyplot.show()
    
    imageio.imwrite('out1.png', mask)
    with open("out1.png", "rb") as image_file:
        out1 = base64.b64encode(image_file.read())
        
    imageio.imwrite('out2.png', buster_mask)
    with open("out2.png", "rb") as image_file:
        out2 = base64.b64encode(image_file.read())
    
    response = {
        'mask': out1.decode('utf-8'),
        'buster_mask': out2.decode('utf-8'),
        'ptime': ptime,
        'btime': btime
    }
        
    return jsonify(response)

def buster_pred(rgb, busterNetModel):
    t0 = datetime.now()
    with graph.as_default():
        pred = simple_cmfd_decoder( busterNetModel, rgb )
    t1 = datetime.now()
    return pred, (t1-t0).total_seconds()

#ManTraNet Utils
def read_rgb_image( image_file ) :
    rgb = cv2.imread( image_file, 1 )[...,::-1]
    return rgb
    
def decode_an_image_array( rgb, manTraNet ) :
    x = np.expand_dims( rgb.astype('float32')/255.*2-1, axis=0 )
    t0 = datetime.now()
    with graph.as_default():
        y = manTraNet.predict(x)[0,...,0]
    t1 = datetime.now()
    return y, t1-t0

def decode_an_image_file( image_file, manTraNet ) :
    rgb = read_rgb_image( image_file )
    rgb = cv2.resize(rgb, dsize=(640,480), interpolation=cv2.INTER_CUBIC)
    mask, ptime = decode_an_image_array( rgb, manTraNet )
    return rgb, mask, ptime.total_seconds()

def init():
    manTraNet = modelCore.load_pretrain_model_by_index( 4, manTraNet_modelDir )
    return manTraNet

if __name__ == '__main__':
    app.run()
























