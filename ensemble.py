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

use_cuda = torch.cuda.is_available()

def evaluate_ensemble_model(opts, models, loader, criterion, l1_criterion):
    for model in models:
        model.eval()

    time_start = time.time()
    val_loss = 0.0
    val_l1_norm = 0.0
    num_batches = 0.0

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

            l1_norm = l1_criterion(pred_year, gt_year)

            val_loss += loss.data.cpu().item()
            val_l1_norm += l1_norm.data.cpu().item()
            num_batches += 1

    avg_valid_loss = val_loss / num_batches
    avg_l1_norm = val_l1_norm / num_batches
    time_taken = time.time() - time_start

    return avg_valid_loss, avg_l1_norm, time_taken

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

    valid_dataset = YearBookDataset(opts.data_dir, split='valid', nclasses=opts.nclasses, target_type=opts.target_type, \
        img_size=opts.img_size, resize=opts.resize)
    valid_loader = DataLoader(valid_dataset, batch_size=opts.bsize, shuffle=opts.shuffle, num_workers=opts.nworkers, pin_memory=True)

    if opts.target_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    elif opts.target_type == 'regression':
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError('Unknown model target type')

    l1_criterion = nn.L1Loss()

    pdb.set_trace()

    avg_valid_loss, avg_l1_norm, time_taken = evaluate_ensemble_model(opts, models, valid_loader, criterion, l1_criterion)

