# import
import sys
import os
import glob
import math

import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from pyopt import modulate as md


def load_image(target_dir, image_number):
    image_dir = '../data/image/'
    image_path_list = sorted(glob.glob(os.path.join(image_dir + target_dir, '*.jpg')))
    image_path = image_path_list[image_number]
    img = Image.open(image_path)
    return img


def image_correlation(img_array0, img_array1):
    correlation = np.zeros((img_array0.shape[0], img_array0.shape[1]), dtype=float)
    for i in tqdm(range(img_array0.shape[0])):
        for j in range(img_array0.shape[1]):
            xor = 0
            for k in range(3):
                xor += bin(img_array0[i, j, k] ^ img_array1[i, j, k]).count('1')
            correlation[i, j] = 1 - xor / 24
    return correlation


target_dir = 'download'
image_number0 = 0
image_number1 = 1

img0 = load_image(target_dir, image_number0)
img_array0 = np.array(img0)
print(img_array0.shape)
img1 = load_image(target_dir, image_number1)
img_array1 = np.array(img1)
print(img_array1.shape)

fig = plt.figure(figsize=(14, 6))
ax0 = fig.add_subplot(1, 2, 1)
ax1 = fig.add_subplot(1, 2, 2)

ax0.imshow(img0)
ax1.imshow(img1)

plt.subplots_adjust(left=0.05, bottom=0, right=0.99, top=0.99, wspace=0.10, hspace=0.20)

corr = image_correlation(img_array0[:min(img_array0.shape[0], img_array1.shape[0]), :min(img_array0.shape[1], img_array1.shape[1])], img_array1[:min(img_array0.shape[0], img_array1.shape[0]), :min(img_array0.shape[1], img_array1.shape[1])])
print(np.mean(corr))
