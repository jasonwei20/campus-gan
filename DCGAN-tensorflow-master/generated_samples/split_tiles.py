import os, sys
from os import listdir
from os.path import isfile, join, isdir

import numpy as np
from scipy.misc import imsave
from PIL import Image
Image.MAX_IMAGE_PIXELS=1e10
from random import randint
import time
from scipy.stats import mode
import cv2
import openslide

import skimage.measure
from skimage.transform import rescale, rotate
import time


input_image = 'train_270_0008.png'
output_folder = 'gen_images'

def output_tiles(input_image, output_folder):
	composite = Image.open(input_image)
	tile_size = 64

	for i in range(8):
		for j in range(8):
			x = i*tile_size
			y = j*tile_size
			tile = composite.crop((x, y, x+tile_size, y+tile_size))
			tile.save(join(output_folder, str(i)+str(j)+".jpg"))

output_tiles(input_image, output_folder)



