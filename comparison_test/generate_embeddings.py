import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pickle
import time
import os
from os import listdir
from os.path import isfile, join, isdir

pic_one = str("eleven_campuses_64/0.jpg")

mylayers = ['layer1', 'layer2', 'layer3', 'layer4']
get_size = {"layer1": 56, "layer2": 28, "layer3": 14, "layer4": 7}
get_depth = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}

def get_image_paths(folder):
	image_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
	if join(folder, '.DS_Store') in image_paths:
		image_paths.remove(join(folder, '.DS_Store'))
	return image_paths

def get_vector(image_name, model, mylayer):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(1, get_depth[mylayer], get_size[mylayer], get_size[mylayer])
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    embedding = my_embedding.numpy()
    embedding = np.mean(embedding, axis = 3)
    embedding = np.mean(embedding, axis = 2)
    embedding = np.mean(embedding, axis = 0)

    return embedding #.view(512).numpy()




for mylayer in mylayers:

	print(mylayer)
	model = models.resnet18(pretrained=True)
	layer = model._modules.get(mylayer)
	model.eval()
	scaler = transforms.Resize((224, 224))
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	to_tensor = transforms.ToTensor()

	embeddings_for_layer = np.zeros((128, get_depth[mylayer])) #first 64 are real, last 64 are generated images
	index = 0

	for folder in ['eleven_campuses_64', 'gen_images']:
		image_paths = get_image_paths(folder)
		for image_path in image_paths:
			embedding = get_vector(image_path, model, mylayer)
			embeddings_for_layer[index, :] = embedding
			print(mylayer, image_path, embedding.shape)
			index += 1

	print(embeddings_for_layer[:5, :5], embeddings_for_layer[-5:, -5:])
	pickle.dump( embeddings_for_layer, open( "embeddings_layer_"+mylayer[-1]+".p", "wb" ) )



	

	


