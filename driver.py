import os
import pdb
import time
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

from models.resnet import ResNet, BasicBlock
from models.alexnet import AlexNet
from utils.dataset import YearBookDataset
from utils.misc import set_random_seeds, get_class_counts
from utils.arguments import get_args

use_cuda = torch.cuda.is_available()

def plot_class_hist(opts):
    cnts = get_class_counts(opts)
    print (cnts)

def evaluate_model(opts, model, loader, criterion):
    model.eval()

    time_start = time.time()
    val_loss = 0.0
    num_batches = 0.0

    for i, d in enumerate(loader):
        if opts.target_type == 'regression':
            d['label'] = d['label'].float()
            d['label'] = d['label'].unsqueeze(1)

        if use_cuda:
            d['image'] = d['image'].cuda()
            d['label'] = d['label'].cuda()

        output = model(d['image'])
        loss = criterion(output, d['label'])

        val_loss += loss.data.cpu().item()
        num_batches += 1

    avg_valid_loss = val_loss / num_batches
    time_taken = time.time() - time_start

    return avg_valid_loss, time_taken

def train(opts):

    train_dataset = YearBookDataset(opts.data_dir, split='train')
    valid_dataset = YearBookDataset(opts.data_dir, split='valid')

    train_loader = DataLoader(train_dataset, batch_size=opts.bsize, shuffle=opts.shuffle, num_workers=opts.nworkers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=opts.bsize, shuffle=opts.shuffle, num_workers=opts.nworkers, pin_memory=True)

    if opts.arch == 'resnet18':
        model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=opts.nclasses, mode=opts.target_type)
    elif opts.arch == 'resnet34':
        model = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=opts.nclasses, mode=opts.target_type)
    elif opts.arch == 'alexnet':
        model = AlexNet(num_classes=opts.nclasses, mode=opts.target_type)
    else:
        raise NotImplementedError('Unsupported model architecture')

    if opts.target_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    elif opts.target_type == 'regression':
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError('Unknown model target type')

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    if opts.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), opts.lr, weight_decay=opts.wd)
    elif opts.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), opts.lr, momentum=opts.momentum, weight_decay=opts.wd)
    else:
        raise NotImplementedError('Unknown optimizer type')

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_gamma)

    # for logging
    n_iter = 0
    writer = SummaryWriter(log_dir=opts.log_dir)
    loss_log = {'train/loss' : 0.0}
    time_start = time.time()
    num_batches = 0

    # for choosing the best model
    best_val_loss = float('inf')

    for epoch in range(opts.start_epoch, opts.epochs):
        model.train()
        scheduler.step()
        for i, d in enumerate(train_loader):
            if opts.target_type == 'regression':
                d['label'] = d['label'].float()
                d['label'] = d['label'].unsqueeze(1)

            if use_cuda:
                d['image'] = d['image'].cuda()
                d['label'] = d['label'].cuda()

            output = model(d['image'])
            loss = criterion(output, d['label'])

            # perform update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), opts.max_norm)
            optimizer.step()

            # log the losses
            n_iter += 1
            num_batches += 1
            loss_log['train/loss'] += loss.data.cpu().item()

            if num_batches != 0 and n_iter % opts.log_iter == 0:
                time_end = time.time()
                time_taken = time_end - time_start
                avg_train_loss = loss_log['train/loss'] / num_batches

                print ("epoch: %d, updates: %d, time: %.2f, avg_train_loss: %.5f" % (epoch, n_iter, time_taken, avg_train_loss))
                # writing values to SummaryWriter
                writer.add_scalar('train/loss', avg_train_loss, n_iter)
                # reset values back
                loss_log = {'train/loss': 0.0}
                num_batches = 0.0
                time_start = time.time()

        val_loss, time_taken = evaluate_model(opts, model, valid_loader, criterion)
        print ("epoch: %d, updates: %d, time: %.2f, avg_valid_loss: %.5f" % (epoch, n_iter, time_taken, val_loss))
        # writing values to SummaryWriter
        writer.add_scalar('val/loss', val_loss, n_iter)
        print ('')

        # Save the model to disk
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            save_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'n_iter': n_iter,
                'opts': opts,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss
            }
            model_path = os.path.join(opts.save_path, 'model_best.net')
            torch.save(save_state, model_path)

        save_state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'n_iter': n_iter,
            'opts': opts,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss
        }
        model_path = os.path.join(opts.save_path, 'model_latest.net')
        torch.save(save_state, model_path)


if __name__ == '__main__':

    opts = get_args()
    set_random_seeds(opts.seed)

    if opts.mode == 'train':
        train(opts)
    elif opts.mode == 'class_dist':
        plot_class_hist(opts)
    else:
        raise NotImplementedError('Unrecognised mode')

