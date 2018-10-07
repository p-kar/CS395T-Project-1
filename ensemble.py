import os
import pdb
import time
import glob
import random
import shutil
import warnings
import argparse
import tensorboardX
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader

from models.resnet import *
from models.alexnet import *
from models.xception import *
from models.se_resnet import senet50
from models.loader import load_model
from utils.dataset import YearBookDataset
from utils.misc import set_random_seeds
from utils.arguments import get_args

# from keras.applications.xception import (
#     Xception, preprocess_input, decode_predictions
# )
# from keras.preprocessing import image
# import keras.models as kmodels
# from keras import backend
# import numpy as np

use_cuda = torch.cuda.is_available()

# def l1_norm(y_true, y_pred):
#     year_true = (y_true * 104) + 1905
#     year_pred = (y_pred * 104) + 1905
#     return backend.sqrt(backend.mean(backend.square(year_pred - year_true), axis=-1))

# def get_xception_output(xception_model, data):

#     res = []

#     for path in data['image_name']:
#         img = image.load_img(path, target_size=(299, 299))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)
#         pred = xception_model.predict(x)
#         result = [int(round((x * 104) + 1905)) for x in pred]
#         res.extend(result)

#     ret = torch.FloatTensor(res)
#     if use_cuda:
#         ret = ret.cuda()

#     return ret

def evaluate_ensemble_model(opts, models, loader, criterion, l1_criterion):
    for model in models:
        model.eval()

    time_start = time.time()
    val_loss = 0.0
    val_l1_norm = 0.0
    num_batches = 0.0

    pred_year_lst = []

    with torch.no_grad():
        for i, d in enumerate(loader):

            if use_cuda:
                d['image'] = d['image'].cuda()
                d['label'] = d['label'].cuda()
                d['year'] = d['year'].cuda()

            output = None

            for model in models:
                if output is None:
                    output = model(d['image'])
                else:
                    output += model(d['image'])

            output /= len(models)

            loss = criterion(output, d['label'])

            gt_year = d['year'].float()

            if opts.target_type == 'regression':
                pred_year = torch.round(output * opts.nclasses + loader.dataset.start_date)
                pred_year = pred_year.view(gt_year.shape)
            elif opts.target_type == 'classification':
                _, pred_year = torch.max(output, 1)
                pred_year = pred_year.float() + loader.dataset.start_date
                pred_year = pred_year.view(gt_year.shape)

            pred_year_lst.extend(pred_year.cpu().numpy().astype(int).tolist())

            # xception_out = get_xception_output(xception_model, d)
            # xception_out.view(gt_year.shape)

            # pred_year = (pred_year + xception_out) / 2.0

            l1_norm = l1_criterion(pred_year, gt_year)

            val_loss += loss.data.cpu().item()
            val_l1_norm += l1_norm.data.cpu().item()
            num_batches += 1

    avg_valid_loss = val_loss / num_batches
    avg_l1_norm = val_l1_norm / num_batches
    time_taken = time.time() - time_start

    return avg_valid_loss, avg_l1_norm, time_taken, pred_year_lst

if __name__ == '__main__':

    opts = get_args()
    set_random_seeds(opts.seed)

    trained_model_paths = [f for f in os.listdir(opts.ensemble_dir) if f.endswith('.net')]

    models = []
    for path in trained_model_paths:
        if not use_cuda:
            models.append(load_model(os.path.join(opts.ensemble_dir, path)))
        else:
            models.append(load_model(os.path.join(opts.ensemble_dir, path)).cuda())

    # xception_model = kmodels.load_model(os.path.join(opts.ensemble_dir, 'xception.h5'), custom_objects={'l1_norm': l1_norm})

    valid_dataset = YearBookDataset(opts.data_dir, split='valid', nclasses=opts.nclasses, target_type=opts.target_type, \
        img_size=opts.img_size, resize=opts.resize)
    valid_loader = DataLoader(valid_dataset, batch_size=opts.bsize, shuffle=False, num_workers=1, pin_memory=True)

    if opts.target_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    elif opts.target_type == 'regression':
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError('Unknown model target type')

    l1_criterion = nn.L1Loss()

    avg_valid_loss, avg_l1_norm, time_taken, pred_years = evaluate_ensemble_model(opts, models, valid_loader, criterion, l1_criterion)

    print ("time: %.2f, avg_valid_loss: %.5f, avg_valid_l1_norm: %.5f" % (time_taken, avg_valid_loss, avg_l1_norm))

    with open('predictions.txt', 'w') as f:
        for year in pred_years:
            f.write('%d\n' % year)

