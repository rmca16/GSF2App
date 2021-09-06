

import sys
import numpy as np
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
def add_cnn_linear_layers(network, pretrained_model, last_out_features, num_classes):
    # Change network's last layer feature map
    if network == "densenet":
        pretrained_model.classifier = nn.Linear(in_features = last_out_features, out_features = 512, bias = True)
   
    elif network == "mobilenet_v2":
        pretrained_model.classifier[0] = nn.Linear(in_features = last_out_features, out_features = 512, bias = True)
        pretrained_model.classifier = nn.Sequential(*list(pretrained_model.classifier.children())[:-1])
    
    else:
        pretrained_model.fc = nn.Linear(in_features = last_out_features, out_features = 512, bias = True)
    
    # Add classifier stage
    pretrained_model = nn.Sequential(pretrained_model,
            #nn.Linear(in_features = last_out_features, out_features = 512, bias = True),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.5),
            nn.Linear(in_features = 512, out_features = 256, bias = True),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.5),
            nn.Linear(in_features = 256, out_features = num_classes, bias = True))
    
    return pretrained_model


def load_pre_trained_model(network, num_classes, learning_stage, model_path):
    if network == "vgg16" or network == "vgg19":
        if network == "vgg16":
            pretrained_model = models.vgg16_bn(pretrained = True)
        elif network == "vgg19":
            pretrained_model = models.vgg19_bn(pretrained = True)
        pretrained_model.classifier[0] = nn.Linear(in_features = 25088, out_features = 512, bias = True)
        pretrained_model.classifier[3] = nn.Linear(in_features = 512, out_features = 256, bias = True)
        pretrained_model.classifier[6] = nn.Linear(in_features = 256, out_features = num_classes, bias = True)

    elif network == "resnet18":
        pretrained_model = models.resnet18(pretrained = True)
        pretrained_model = add_cnn_linear_layers(network, pretrained_model, 512, num_classes)

    elif network == "resnet50":
        pretrained_model = models.resnet50(pretrained = True)
        pretrained_model = add_cnn_linear_layers(network, pretrained_model, 2048, num_classes)

    elif network == "resnet101":
        pretrained_model = models.resnet101(pretrained = True)
        pretrained_model = add_cnn_linear_layers(network, pretrained_model, 2048, num_classes)

    elif network == "densenet":
        pretrained_model = models.densenet121(pretrained = True)
        pretrained_model = add_cnn_linear_layers(network, pretrained_model, 1024, num_classes)

    elif network == "mobilenet_v2":
        pretrained_model = models.mobilenet_v2(pretrained = True)
        pretrained_model = add_cnn_linear_layers(network, pretrained_model, 1280, num_classes)


    if learning_stage == 2:
        pretrained_model.load_state_dict(torch.load(model_path))
        set_parameter_requires_grad(pretrained_model, True)

        if network == "vgg16" or network == "vgg19":
            pretrained_model.classifier = nn.Sequential(*list(pretrained_model.classifier.children())[:-1])  #Last layer Removed
        else:
            pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])

    return pretrained_model

# ------------------------------------------------- #
# ---------------------- GSF2AppV2 ---------------------- #
class GSF2AppV2_correlation(nn.Module):
    def __init__(self, network, learning_stage, cnn_model_path, num_classes, k):
        super(GSF2AppV2_correlation, self).__init__()
        self.num_classes = num_classes
        self.learning_stage = learning_stage
        self.k = k

        # -------------- CNN -------------- #
        self.network = network
        self.cnn_model_path = cnn_model_path
        self.cnn_model = load_pre_trained_model(self.network, self.num_classes,
            self.learning_stage, self.cnn_model_path)

        # -------------- SFV -------------- #
        self.sfv_conv1 = nn.Conv1d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.sfv_conv2 = nn.Conv1d(in_channels = 32, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.sfv_conv3 = nn.Conv1d(in_channels = 128, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)

        self.sfv_fc1 = nn.Linear(in_features = 32*80*1, out_features = 1024, bias = True)

        # -------------- SF3M -------------- #
        self.sfm_conv1 = nn.Conv2d(in_channels = self.k, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.sfm_conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.sfm_conv3 = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)

        self.sfm_fc1 = nn.Linear(in_features = 64*40*40, out_features = 1024, bias = True)

        # -------------- Feature Fusion -------------- #
        self.fus2_fc1 = nn.Linear(in_features = 897, out_features = 512)

        # -------------- Network Output -------------- #
        self.fus_out = nn.Linear(in_features = 512, out_features = self.num_classes)


    def forward(self, imgs, sfv_x, sfm_x):
        # -------------- CNN -------------- #
        x_cnn = self.cnn_model(imgs)
        # -------------- SFV -------------- #
        x_sfv = F.relu(self.sfv_conv1(sfv_x))
        x_sfv = F.relu(self.sfv_conv2(x_sfv))
        x_sfv = F.relu(self.sfv_conv3(x_sfv))

        x_sfv = x_sfv.view(-1, 32*80*1)
        x_sfv = self.sfv_fc1(x_sfv)

        # -------------- SF3M -------------- #
        x_sfm = F.relu(self.sfm_conv1(sfm_x))
        x_sfm = F.max_pool2d(x_sfm, kernel_size = 3, stride = 2, padding = 1)
        x_sfm = F.relu(self.sfm_conv2(x_sfm))
        x_sfm = F.relu(self.sfm_conv3(x_sfm))

        x_sfm = x_sfm.view(-1, 64*40*40)
        x_sfm = self.sfm_fc1(x_sfm)

        # -------------- Feature Fusion -------------- #
        x_fus = torch.cat((x_sfv, x_sfm), dim = 1)
        x_fus = self.custom_conv1d_features_np(x_cnn, x_fus, stride = 2)
        x_fus = F.relu(self.fus2_fc1(x_fus))
        x_out = self.fus_out(x_fus)     

        return x_out


    def custom_conv1d_features_np(self, kernel, features, stride):
        kernel_np = kernel.cpu().detach().numpy()
        features_np = features.cpu().detach().numpy()
        batch_len = features_np.shape[0]; f_len = features_np.shape[1]; k_len = kernel_np.shape[1]
        number_out_features = math.floor((f_len-k_len)/stride)+1
        out_features = np.zeros((batch_len, number_out_features)) # [32, 257]

        for b in range(0, batch_len): # batch
            for z in range(0, out_features.shape[1]): # size of output feature map
                out_features[b,z] = sum(kernel_np[b,:]*features_np[b,z*stride:z*stride+k_len])
        out_features = torch.from_numpy(out_features).float()
        out_features = out_features.to("cuda:0")
        return out_features




class GSF2AppV2_concat(nn.Module):
    def __init__(self, network, learning_stage, cnn_model_path, num_classes, k):
        super(GSF2AppV2_concat, self).__init__()
        self.num_classes = num_classes
        self.learning_stage = learning_stage
        self.k = k

        # -------------- CNN -------------- #
        self.network = network
        self.cnn_model_path = cnn_model_path
        self.cnn_model = load_pre_trained_model(self.network, self.num_classes,
            self.learning_stage, self.cnn_model_path)

        # -------------- SFV -------------- #
        self.sfv_conv1 = nn.Conv1d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.sfv_conv2 = nn.Conv1d(in_channels = 32, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.sfv_conv3 = nn.Conv1d(in_channels = 128, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.sfv_fc1 = nn.Linear(in_features = 32*80*1, out_features = 1024, bias = True)

        # -------------- SF3M -------------- #
        self.sfm_conv1 = nn.Conv2d(in_channels = self.k, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.sfm_conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.sfm_conv3 = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.sfm_fc1 = nn.Linear(in_features = 64*40*40, out_features = 1024, bias = True)

        # -------------- Feature Fusion -------------- #
        self.fus2_fc1 = nn.Linear(in_features = 256+1024+1024, out_features = 512)
        # -------------- Network Output -------------- #
        self.fus_out = nn.Linear(in_features = 512, out_features = self.num_classes)


    def forward(self, imgs, sfv_x, sfm_x):
        # -------------- CNN -------------- #
        x_cnn = self.cnn_model(imgs)
        # -------------- SFV -------------- #
        x_sfv = F.relu(self.sfv_conv1(sfv_x))
        x_sfv = F.relu(self.sfv_conv2(x_sfv))
        x_sfv = F.relu(self.sfv_conv3(x_sfv))

        x_sfv = x_sfv.view(-1, 32*80*1)
        x_sfv = self.sfv_fc1(x_sfv)
        # -------------- SF3M -------------- #
        x_sfm = F.relu(self.sfm_conv1(sfm_x))
        x_sfm = F.max_pool2d(x_sfm, kernel_size = 3, stride = 2, padding = 1)
        x_sfm = F.relu(self.sfm_conv2(x_sfm))
        x_sfm = F.relu(self.sfm_conv3(x_sfm))

        x_sfm = x_sfm.view(-1, 64*40*40)
        x_sfm = self.sfm_fc1(x_sfm)

        # -------------- Feature Fusion -------------- #
        x_fus = torch.cat((x_cnn, x_sfv, x_sfm), dim = 1)
        x_fus = F.relu(self.fus2_fc1(x_fus))
        x_out = self.fus_out(x_fus)     

        return x_out

# ------------------------------------------------- #



if __name__ == '__main__':
    print("PyTorch version = {}".format(torch.__version__))

   
    cnn_model_path = '../fake_path.pth'

    GSF2AppV2_model = GSF2AppV2_concat("vgg16", 2, cnn_model_path, 10, 3)





