###################################################################################
# Author: Ricardo Pereira
# Date: 06-06-2021
# Last Modified data: 10-09-2021
# Abstract: GSF2App: Dataset management file
###################################################################################

import numpy as np
import time
import glob
import cv2
import os
import torch
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

from YOLOv3.models import *
from YOLOv3.utils import *


class Dataset(data.Dataset):
    def __init__(self, data_path):
        'Initialization'

        self.training_stage = 1
        self.imgs_path = glob.glob(data_path+'*.png')
        if not self.imgs_path:
            print("No png images available")
            sys.exit()

        self.yolo = YOLOv3()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.imgs_path)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        img_path = self.imgs_path[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224), fx = 1, fy = 1, interpolation = cv2.INTER_LINEAR)

        # Load data and get label
        X = torch.from_numpy(img).float()

        X_data = X.permute(2,0,1)

        y = (np.long(0))
        print("PLEASE, ADD SOME CODE TO READ LABELS FROM DE DATASET")

        if self.training_stage == 1:
            return X_data, y

        elif self.training_stage == 2:
        	yolo_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            sfv = self.yolo.get_semantic_features(yolo_img)
            sfv = sfv.astype(np.float16)
            sfv = torch.from_numpy(sfv).float()

            return X_data, sfv, y




class YOLOv3(object):
	def __init__(self):
		self.model_def	  = os.path.join('YOLOv3','config','yolov3.cfg')
		self.weights_path   = os.path.join('YOLOv3','weights','yolov3.weights')
		self.class_path	 = os.path.join('YOLOv3','weights','coco.names')
		self.conf_thres	 = 0.2
		self.nms_thres	  = 0.45
		self.img_size	   = 416

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Set up model
		self.model = Darknet(self.model_def, img_size = self.img_size).to(self.device)
		if self.weights_path.endswith(".weights"):
			self.model.load_darknet_weights(self.weights_path)
		else:
			self.model.load_state_dict(torch.load(self.weights_path))
		self.model.eval()  # Set in evaluation mode

		self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

		self.yolo_classes = load_classes(self.class_path)


	def get_semantic_features(self, image):
		img_shape = image.shape

		# Extract image as PyTorch tensor
		img = transforms.ToTensor()(image)
		# Pad to square resolution
		img, _ = pad_to_square(img, 0)
		# Resize
		img = resize(img, self.img_size)
		# Configure input
		input_img = Variable(img.type(self.Tensor))

		# Get detections
		with torch.no_grad():
			#print(input_img.unsqueeze(0).shape)
			detections = self.model(input_img.unsqueeze(0))
			detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)

		sfv = np.zeros((len(self.yolo_classes)), dtype = np.int8)

		for idx, img_detections in enumerate(detections):  
			if img_detections is not None:
				# Rescale boxes to original image
				img_detections = rescale_boxes(img_detections, self.img_size, img_shape[:2])
				unique_labels = img_detections[:, -1].cpu().unique()

				for x1, y1, x2, y2, conf, cls_conf, cls_pred in img_detections:
					sfv[int(cls_pred)] += 1

		return sfv