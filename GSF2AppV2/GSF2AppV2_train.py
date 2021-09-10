###################################################################################
# Author: Ricardo Pereira
# Date: 11-07-2021
# Last Modified data: 13-07-2021
# Abstract: GSF2AppV2: Training file
###################################################################################

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
import argparse
import torchsummary
import sys

from GSF2AppV2_data import *
from GSF2AppV2_network import *
from GSF2AppV2_eval import *  


def check_GPU():
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print ("\nDevice: " + str(device))
	return device



def training_GSF2AppV2(trainloader, network, learning_rate, classes, cnn_model_path, gsf2app_model_path, n_epochs, training_stage, fusion_method):
	# Initializations
	device = check_GPU()

	if training_stage == 1:
		model = load_pre_trained_model(network, len(classes), training_stage, cnn_model_path)

	elif training_stage == 2:
		if fusion_method == "concat":
			model = GSF2AppV2_concat(network, training_stage, cnn_model_path, len(classes), 3)
		else:
			model = GSF2AppV2_correlation(network, training_stage, cnn_model_path, len(classes), 3)


	model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0.0005)
	#optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum =0.9, weight_decay = 0.0005)

	print("Training...")
	for epoch in range(n_epochs):
		for local_batch, data in enumerate(trainloader, 0):
			if training_stage == 1:
				images, labels = data
				images, labels = images.to(device), labels.to(device)
				outputs = model(images)

			elif training_stage == 2:
				images, sfv_data, sfm_data, labels = data
				images, sfv_data, sfm_data, labels = images.to(device), sfv_data.to(device), sfm_data.to(device), labels.to(device)
				outputs = model(images, sfv_data, sfm_data)

			optimizer.zero_grad()
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print loss per batch
			if local_batch == 0:
				print ('[%d, %5d] loss: %.3f' % (epoch + 1, local_batch + 1,  loss.item()))

	# Save the model
	if training_stage == 1:
		torch.save(model.state_dict(), cnn_model_path)
		print('\nGSF2AppV2\'s learning stage ' + str(training_stage) + ' was successfully trained!')
		print('Model saved in: ' + cnn_model_path)
	elif training_stage == 2:
		torch.save(model.state_dict(), gsf2app_model_path)
		print('\nGSF2AppV2\'s learning stage ' + str(training_stage) + ' was successfully trained!')
		print('Model saved in: ' + gsf2app_model_path)






if __name__ == '__main__':
	print("PyTorch version = {}".format(torch.__version__))

	parser = argparse.ArgumentParser()
	parser.add_argument("--stage_1_n_epochs", type = int, default=1, help="stage 1 number of epochs") # 75
	parser.add_argument("--stage_1_lr", type=float, default=0.0001, help="stage 1 learning rate")
	parser.add_argument("--stage_2_n_epochs", type = int, default=1, help="stage 2 number of epochs") # 25
	parser.add_argument("--stage_2_lr", type=float, default=0.0001, help="stage 2 learning rate")
	parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
	parser.add_argument("--dataset", type=str, default="nyu", help="dataset") #nyu or sun
	parser.add_argument("--backbone", type=str, default="vgg16", help= "backbone network")
	parser.add_argument("--sfm_k", type=int, default=3, help= "SFM k")
	parser.add_argument("--fusion_method", type=str, default="concat", help= "feature fusion approach: concat or correlation")
	opt = parser.parse_args()
	print(opt)

	# --------------- Load Dataset --------------- #
	print("\n# ----- " + opt.dataset + " Dataset ----- #")

	X_train_datapath= os.path.join('folder_data_path')
	X_test_datapath= os.path.join('folder_data_path')

	scene_classes = open(os.path.join('folder_path','file.names'), "r").read().split("\n")[:-1]
	print("Number of classes: " + str(len(scene_classes)))

	# --------------- Prepare Dataset --------------- #
	training_set = Dataset(X_train_datapath, opt.sfm_k)
	test_set = Dataset(X_test_datapath, opt.sfm_k)

	train_loader = data.DataLoader(dataset=training_set, batch_size = opt.batch_size, shuffle = False)
	test_loader = data.DataLoader(dataset=test_set, batch_size = opt.batch_size, shuffle = False)

	# --------------- Training --------------- #
	
	if not os.path.isdir(os.path.join('models','cnn',opt.dataset)):
		os.makedirs(os.path.join('models','cnn',opt.dataset))
	if not os.path.isdir(os.path.join('models','gsf2app',opt.dataset)):
		os.makedirs(os.path.join('models','gsf2app',opt.dataset))

	cnn_model_path = os.path.join('models','cnn',opt.dataset,opt.backbone+'_'+opt.dataset+'.ckpt')
	gsf2app_model_path = os.path.join('models','gsf2app',opt.dataset,'gsf2app_'+opt.backbone+'_'+opt.dataset+'.ckpt') #str(i)+


	# Train CNN model
	training_GSF2AppV2(train_loader, opt.backbone, opt.stage_1_lr, scene_classes, cnn_model_path,
		gsf2app_model_path, opt.stage_1_n_epochs, 1, opt.fusion_method)

	# Eval CNN model
	eval_GSF2AppV2(test_loader, opt.backbone, scene_classes, cnn_model_path, gsf2app_model_path, 1, opt.fusion_method)

	# Train GSF2AppV2 model
	training_set.training_stage = 2
	training_GSF2AppV2(train_loader, opt.backbone, opt.stage_2_lr, scene_classes, cnn_model_path,
		gsf2app_model_path, opt.stage_2_n_epochs, 2, opt.fusion_method)

	# Eval GSF2AppV2 model
	test_set.training_stage = 2
	eval_GSF2AppV2(test_loader, opt.backbone, scene_classes, cnn_model_path, gsf2app_model_path, 2, opt.fusion_method)