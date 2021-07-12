

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

import time
import math


def set_parameter_requires_grad(model, feature_extracting):
	# This function allows to freeze model's weights
	if feature_extracting:
		for param in model.parameters():
			param.requires_grad = False

# ---------------------- CNN ---------------------- #
def load_pre_trained_model(num_classes, learning_stage, model_path):
	pretrained_model = models.vgg16_bn(pretrained = True)
	pretrained_model.classifier[0] = nn.Linear(in_features = 25088, out_features = 512, bias = True)
	pretrained_model.classifier[3] = nn.Linear(in_features = 512, out_features = 256, bias = True)
	pretrained_model.classifier[6] = nn.Linear(in_features = 256, out_features = num_classes, bias = True)

	if learning_stage == 2:
		pretrained_model.load_state_dict(torch.load(model_path))
		set_parameter_requires_grad(pretrained_model, True)

		#Remove Last layer
		pretrained_model.classifier = nn.Sequential(*list(pretrained_model.classifier.children())[:-1])

	return pretrained_model

# ------------------------------------------------- #
# ---------------------- GSF2App ---------------------- #
class GSF2App(nn.Module):
	def __init__(self, cnn_model_path, num_classes):
		super(GSF2App, self).__init__()
		self.num_classes	= num_classes
		self.cnn_model_path = cnn_model_path

		# ---------- CNN  ---------- #
		self.cnn_model = load_pre_trained_model(self.num_classes, 2, self.cnn_model_path)

		# ---------- SFV NN  ---------- #
		self.sfv_fc4 = nn.Linear(in_features = 80, out_features = 1024, bias = True)
		self.sfv_fc5 = nn.Linear(in_features = 1024, out_features = 512, bias = True)

		# ---------- Fusion  ---------- #
		self.fus_fc1 = nn.Linear(in_features = 256 + 512, out_features = 256, bias = True)
		self.fus_fc6 = nn.Linear(in_features = 256, out_features = self.num_classes, bias = True)


	def forward(self, imgs, sfv):
		# ---------- VGG16  ---------- #
		x_cnn = self.cnn_model(imgs)

		# ---------- SFV NN  ---------- #
		x_sfv = F.relu(self.sfv_fc4(sfv))
		x_sfv = self.sfv_fc5(x_sfv)

		# ---------- Fusion  ---------- #
		x_fus = torch.cat((x_cnn, x_sfv), dim = 1)
		x_fus = F.relu(self.fus_fc1(x_fus))
		x_fus = self.fus_fc6(x_fus)

		return(x_fus)



if __name__ == '__main__':
	print("PyTorch version = {}".format(torch.__version__))

	cnn_model_path = '../fake_path.pth'

	gsf2app_model = GSF2App(cnn_model_path, 10)

