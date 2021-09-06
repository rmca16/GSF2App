import numpy as np
import time
import glob
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import argparse

from GSF2AppV2_data import *
from GSF2AppV2_network import * 


def eval_GSF2AppV2(testloader, network, classes, cnn_model_path, gsf2app_model_path, eval_stage, fusion_method):
	n_classes = len(classes)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Define and Load model
	if eval_stage == 1:
		model = load_pre_trained_model(network, len(classes), eval_stage, cnn_model_path)
		model.load_state_dict(torch.load(cnn_model_path))
		print("\nEvaluating CNN...")

	elif eval_stage == 2:
		if fusion_method == "concat":
			model = GSF2AppV2_concat(network, eval_stage, cnn_model_path, len(classes), 3)
		else:
			model = GSF2AppV2_correlation(network, eval_stage, cnn_model_path, len(classes), 3)
		print("\nEvaluating GSF2AppV2...")

	model.to(device)
	model.eval()

	correct = 0
	total = 0
	class_correct = list(0. for i in range(n_classes))
	class_total = list(0. for i in range(n_classes))
	conf_matrix =[[0 for x in range( n_classes )] for y in range( n_classes )]

	with torch.no_grad():
		for _, data in enumerate(testloader, 0):
			# Get model's prediction
			if eval_stage == 1:
				images, labels = data
				images, labels = images.to(device), labels.to(device)
				outputs = model(images)

			elif eval_stage == 2:
				images, sfv_data, sfm_data, labels = data
				images, sfv_data, sfm_data, labels = images.to(device), sfv_data.to(device), sfm_data.to(device), labels.to(device)
				outputs = model(images, sfv_data, sfm_data)

			_, predicted = torch.max(outputs.data, 1)

			# Performance
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			c = (predicted == labels).squeeze()

			for i in range(len(images)):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1

			conf_matrix+=confusion_matrix(predicted.cpu(), labels.cpu(),labels=[x for x in range(n_classes)])

	# Print the achieved performance
	print('Test Accuracy of the model on the {} test images: {} %'.format(total,100 * correct / total))
	print(conf_matrix)

	for i in range(n_classes):
		if class_total[i] == 0:
			print('Accuracy of %5s : %2d %% in %d Images' % (classes[i], 0, 0))
		else:
			print('Accuracy of %5s : %2d %% in %d Images' % (classes[i], 100 * class_correct[i] / class_total[i], class_total[i]))




if __name__ == '__main__':
	print("PyTorch version = {}".format(torch.__version__))

	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
	parser.add_argument("--dataset", type=str, default="nyu", help="dataset (nyu or sun)")
	opt = parser.parse_args()
	
	# --------------- Model Paths --------------- #
	cnn_model_path = os.path.join('models','cnn',opt.dataset,opt.dataset+'.ckpt')
	gsf2app_model_path = os.path.join('models','gsf2app',opt.dataset,'gsf2app_'+opt.dataset+'.ckpt')

	# --------------- Load Dataset --------------- #
	X_rgb_test= os.path.join('folder_data_path')
	y_test= os.path.join('folder_data_path')


	# --------------- Prepare Dataset --------------- #
	scene_classes = open(os.path.join('folder_path','file.names'), "r").read().split("\n")[:-1]
	test_set = Dataset(X_rgb_test, y_test)
	test_loader = data.DataLoader(dataset=test_set, batch_size = opt.batch_size, shuffle = False)

	# --------------- Eval --------------- #
	# CNN
	eval_gsf2app(test_loader, scene_classes, cnn_model_path, gsf2app_model_path, 1)

	# GSF2App
	test_set.training_stage = 2
	eval_gsf2app(test_loader, scene_classes, cnn_model_path, gsf2app_model_path, 2)



