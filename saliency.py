import os
import pdb
import cv2
import time
import glob
import random
import shutil
import warnings
import argparse
import tensorboardX
from tensorboardX import SummaryWriter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader
from PIL import Image

from models.resnet import *
from models.alexnet import *
from models.xception import *
from models.se_resnet import senet50
from models.loader import load_model
from utils.dataset import YearBookDataset
from utils.misc import set_random_seeds
from utils.arguments import get_args

from torchvision import transforms
from torchvision.transforms import CenterCrop, RandomCrop, ToTensor, RandomHorizontalFlip, Scale

use_cuda = torch.cuda.is_available()

def preprocess_image(img_path, img_label):
	img_name = img_path
	year_label = img_label
	img_label = img_label - 1900
	
	img = Image.open(img_name)
	transform = transforms.Compose([CenterCrop(160), ToTensor()])
	img = transform(img)

	return {'image': img, 'image_name': img_name, 'label': img_label, 'year': year_label}

if __name__ == '__main__':

    opts = get_args()
    set_random_seeds(opts.seed)

    model = load_model(opts.load_model)

    img_path = './data/yearbook/valid/F/001224.png'

    d = preprocess_image(img_path, 2001)

    saliency_map = -1.0 * model.saliency(d['image'].unsqueeze(0), d['label'])
    saliency_map = saliency_map.detach().cpu().numpy()

    saliency_map = (saliency_map - np.min(saliency_map))
    saliency_map /= np.max(saliency_map)

    cam = saliency_map
    cam = cv2.resize(cam, (160, 160))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.65)] = 0
    # heatmap = np.moveaxis(heatmap, -1, 0)
    heatmap = np.power(heatmap, 0.5)
    heatmap = heatmap / np.max(heatmap) * 255
    img = heatmap * 0.5 + 255 * np.moveaxis(d['image'].cpu().numpy(), 0, -1)

    img = (img / np.max(img)) * 255

    cv2.imwrite('saliency.png', img)



