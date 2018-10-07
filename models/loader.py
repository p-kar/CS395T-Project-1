import pdb
import torch
import torch.nn as nn
import math

from .resnet import *

def load_model(model_path, target_type='regression', num_classes=120, img_size=160):
	state_dict = torch.load(model_path, map_location=torch.device('cpu'))
	model_type = state_dict['opts'].arch
	target_type = state_dict['opts'].target_type
	num_classes = state_dict['opts'].nclasses
	img_size = state_dict['opts'].img_size

	model = globals()[model_type](False, target_type=target_type, num_classes=num_classes, img_size=img_size)
	model.load_state_dict(state_dict['state_dict'])

	return model
