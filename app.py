import flask
from flask import Flask, render_template, jsonify
import tempfile, os, glob, re
import json 
from keras.preprocessing.image import load_img 
import requests
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
app = Flask(__name__)
import urllib.request
import json
import os
import ssl
import albumentations as A
A.__version__

batch_size = 20
data_sample = 3000
steps = data_sample//batch_size
img_height, img_width = 256, 256
x_min = (128 - img_width) // 2
y_min = (128 - img_height) // 2


@app.route("/")
def home():
    return render_template('home.html',
    url='/static/image_decoration.jpg')
   
@app.route("/presentation")
def presentation():
    return render_template('presentation.html',
    url='/static/image_decoration.jpg',
    url1='/static/image/aachen_000040_000019.png',
    url2='/static/image/bochum_000000_020673.png',
    url3='static/image/bremen_000177_000019.png',
    url4='static/image/cologne_000027_000019.png')

@app.route("/presentation_fpn")
def presentation_fpn():
    return render_template('presentation_fpn.html',
    url='/static/image_decoration.jpg',
    url1='/static/image/aachen_000040_000019.png',
    url2='/static/image/bochum_000000_020673.png',
    url3='static/image/bremen_000177_000019.png',
    url4='static/image/cologne_000027_000019.png')


@app.route("/model")
def model():
    return render_template('model.html',
    url='/static/image_decoration.jpg',
    url1='/static/unet.png',
    url2='/static/fpn.png')
    

@app.route('/<int:image_id>')
def prediction(image_id):
    batch_size = 20
    data_sample = 3000
    steps = data_sample//batch_size
    img_height, img_width = 256, 256
    x_min = (128 - img_width) // 2
    y_min = (128 - img_height) // 2
    image_list, mask_list = listImg_Mask('static/image','static/mask')
    it = img_to_array(load_img(f'{image_list[image_id]}', target_size=(img_height, img_width)))/255.
    it_comp = img_to_array(load_img(f'{mask_list[image_id]}', target_size=(img_height, img_width), color_mode='grayscale'))/255.
    trs = get_validation_augmentation() #training_augmentation()
    sample = trs(image=it)
    msk = trs(image=it_comp)
    test_mask = np.squeeze(it_comp)
    it = sample['image']
    maskit = sample['image']
    dta = {"data": it.tolist()
}

    body = str.encode(json.dumps(dta))

    url = 'http://40.125.114.201:80/api/v1/service/aks-service-appinsights/score'
    api_key = 'ZqWb8Vc4YstSyEeCzhO78mpETSs6pYya' # Replace this with the API key for the web service
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()
        #print(result)
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        #print(error.info())
        #print(json.loads(error.read().decode("utf8", 'ignore')))
    pred = eval(result.decode("utf-8"))
    pred =  json.loads(pred)
    vis = np.array(pred['result'][0])
    print(vis.shape) 
    import matplotlib.pyplot as plt
    plt.imsave('static/prediction/out1.png', vis[:,:,:3], cmap='nipy_spectral_r')
    plt.imsave('static/prediction/out2.png', test_mask, cmap='nipy_spectral_r')
    plt.imsave('static/prediction/out3.png', it)
    return render_template('prediction.html',
        url='/static/image_decoration.jpg',
        url1='/static/prediction/out1.png',
        url2='/static/prediction/out2.png',
        url3='/static/prediction/out3.png')

@app.route('/<int:image_id>')
def prediction_fpn(image_id):
    image_list, mask_list = listImg_Mask('static/image','static/mask')
    it = img_to_array(load_img(f'{image_list[image_id]}', target_size=(img_height, img_width)))/255.
    it_comp = img_to_array(load_img(f'{mask_list[image_id]}', target_size=(img_height, img_width), color_mode='grayscale'))/255.
    trs = get_validation_augmentation() #training_augmentation()
    sample = trs(image=it)
    msk = trs(image=it_comp)
    test_mask = np.squeeze(it_comp)
    it = sample['image']
    maskit = sample['image']
    dta = {"data": it.tolist()
}

    body = str.encode(json.dumps(dta))

    url = 'http://40.125.114.201:80/api/v1/service/aks-service-appinsights/score'
    api_key = 'ZqWb8Vc4YstSyEeCzhO78mpETSs6pYya' # Replace this with the API key for the web service
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()
        #print(result)
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        #print(error.info())
        #print(json.loads(error.read().decode("utf8", 'ignore')))
    pred = eval(result.decode("utf-8"))
    pred =  json.loads(pred)
    vis = np.array(pred['result'][0])
    print(vis.shape) 
    import matplotlib.pyplot as plt
    plt.imsave('static/prediction/out1.png', vis[:,:,:3], cmap='nipy_spectral_r')
    plt.imsave('static/prediction/out2.png', test_mask, cmap='nipy_spectral_r')
    plt.imsave('static/prediction/out3.png', it)
    return render_template('prediction_fpn.html',
        url='/static/image_decoration.jpg',
        url1='/static/prediction/out1.png',
        url2='/static/prediction/out2.png',
        url3='/static/prediction/out3.png')


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)
    



def get_validation_augmentation():
     
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # A.PadIfNeeded(384, 480)
        A.PadIfNeeded(min_height=x_min, min_width=y_min, always_apply=True, border_mode=0),
    ]
    return A.Compose(test_transform)



# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def listImg_Mask(path_image,path_masks):
  image_list = [f for f in glob.iglob(os.path.join(path_image, "*.png"))]
  mask_list = [f for f in glob.iglob(os.path.join(path_masks, "*.png"))]
  image_list.sort()
  mask_list.sort()
 # print(f'. . . . .Number of images: {len(image_list)}\n. . . . .Number of masks: {len(mask_list)}')
  return image_list,mask_list