from keras.preprocessing.image import load_img
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
from numpy import asarray
import glob, os, shutil
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import albumentations as A
A.__version__


batch_size = 20
data_sample = 3000
steps = data_sample//batch_size
img_height, img_width = 256, 256
class_nb = 8

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

x_min = (128 - img_width) // 2
y_min = (128 - img_height) // 2


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

image_list, mask_list = listImg_Mask('static/image','static/mask')

for x in range(len(image_list)):
    _it = img_to_array(load_img(f'{image_list[x]}', target_size=(img_height, img_width)))/255.
    it_comp = img_to_array(load_img(f'{mask_list[x]}', target_size=(img_height, img_width), color_mode='grayscale'))/255.
    trs = get_validation_augmentation() #training_augmentation()
    sample = trs(image=_it)
    msk = trs(image=it_comp)
    test_mask = np.squeeze(it_comp)
    it = sample['image']
    maskit = sample['image']
    dta = {"data": it.tolist()}

#PIL_image = Image.fromarray(np.uint8(it)).convert('RGBA')

#PIL_image = Image.fromarray(it.astype('uint8'), 'RGBA')

#PIL_image.save('static/prediction/out.png')

import matplotlib.pyplot as plt
plt.imsave('static/prediction/out1.png', test_mask, cmap='nipy_spectral_r')

#visualize(img=it,masque=test_mask,images=it)

import urllib.request
import json
import os
import ssl





