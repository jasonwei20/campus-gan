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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift
from sklearn.svm import SVC
from sklearn.decomposition import PCA

def print_tsne(layer_num):


	in_pickle = "embeddings_layer_"+layer_num+".p"
	outpath = "tsne_layer_"+layer_num+".jpg"
	embeddings_vector = pickle.load( open( in_pickle, "rb" ) )
	print('embedding vector shape', embeddings_vector.shape)
	tsne_embedded = TSNE(n_components=2).fit_transform(embeddings_vector)

	real_points = tsne_embedded[:64]
	gen_points = tsne_embedded[-64:]

	plt.plot(real_points[:, 0], real_points[:, 1], 'go')
	plt.plot(gen_points[:, 0], gen_points[:, 1], 'bo')
	plt.title('Layer ' + layer_num + ' Embeddings of College Campus Images')
	plt.xlabel('x (tsne)')
	plt.ylabel('y (tsne)')
	plt.legend(['Embeddings of Real Images', 'Embeddings of Generated Images'], loc='upper right')
	plt.savefig(outpath, dpi=400)
	plt.clf()

def get_k_means_mislabeled(layer_num):

	in_pickle = "embeddings_layer_"+layer_num+".p"
	embeddings_vector = pickle.load( open( in_pickle, "rb" ) )
	embeddings_vector = PCA(n_components=2).fit_transform(embeddings_vector)

	kmeans = KMeans(n_clusters=2, random_state=2).fit(embeddings_vector)
	pred_labels = kmeans.labels_

	incorrect_predictions = 0
	for i in range(len(pred_labels)):
		prediction = pred_labels[i]
		if i < 64 and not prediction == 0:
			incorrect_predictions += 1
		if i >= 64 and not prediction == 1:
			incorrect_predictions += 1

	print(min(incorrect_predictions, 128-incorrect_predictions))

def get_svm_mislabeled(layer_num):

	in_pickle = "embeddings_layer_"+layer_num+".p"
	X = pickle.load( open( in_pickle, "rb" ) )
	X = PCA(n_components=2).fit_transform(X)
	y = np.zeros((128,))
	for i in range(64, 128):
		y[i] = 1

	clf = SVC()
	clf.fit(X, y) 
	pred_labels = clf.predict(X)

	incorrect_predictions = 0
	for i in range(len(pred_labels)):
		prediction = pred_labels[i]
		if i < 64 and not prediction == 0:
			incorrect_predictions += 1
		if i >= 64 and not prediction == 1:
			incorrect_predictions += 1

	print(min(incorrect_predictions, 128-incorrect_predictions))

layer_nums = ['1', '2', '3', '4']
for layer_num in layer_nums:
	print_tsne(layer_num)
	get_k_means_mislabeled(layer_num)
	get_svm_mislabeled(layer_num)



