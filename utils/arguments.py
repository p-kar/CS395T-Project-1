import argparse

def str2bool(t):
    if t.lower() in ['true', 't', '1']:
        return True
    else:
        return False

def get_args():

     parser = argparse.ArgumentParser(description='CS 395T: Age Prediction from Yearbook Photos')

     # Mode
     parser.add_argument('--mode', default='train', type=str, help='mode of the python script')

     # DataLoader
     parser.add_argument('--data_dir', default='data/yearbook', type=str, help='root directory of the dataset')
     parser.add_argument('--nworkers', default=4, type=int, help='number of data loading workers (default: 4)')
     parser.add_argument('--bsize', default=32, type=int, help='mini-batch size (default: 32)')
     parser.add_argument('--shuffle', default='True', type=str2bool, help='shuffle the data?')
     parser.add_argument('--nclasses', default=120, type=int, help='Number of classes in the dataset')

     # Model Parameters
     parser.add_argument('--arch', default='resnet18', type=str, help='model architecture one of [resnet18, resnet34, alexnet]')
     parser.add_argument('--target_type', default='classification', type=str, help='target is classification or regression')

     # Optimization Parameters
     parser.add_argument('--optim', default='adam', type=str, help='Optimizer type')
     parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
     parser.add_argument('--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
     parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
     parser.add_argument('--epochs', default=90, type=int, help='number of total epochs to run')
     parser.add_argument('--max_norm', type=float, default=1, help='Max grad norm')
     parser.add_argument('--lr_decay_step', type=int, default=45, help='learning rate decay step (after how many epochs)')
     parser.add_argument('--lr_decay_gamma', type=float, default=0.1, help='learning rate decay gamma')

     parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')

     # Save Parameter
     parser.add_argument('--save_path', type=str, default='./trained_models', help='Directory where models are saved')

     # Other
     parser.add_argument('--log_dir', default='./logs', help='Directory where tensorboardX logs are saved')
     parser.add_argument('--log_iter', default=10, type=int, help='print frequency (default: 10)')
     parser.add_argument('--resume', default=False, type=str2bool, help='path to latest checkpoint (default: none)')
     parser.add_argument('--seed', default=123, type=int, help='seed for initializing training')

     args = parser.parse_args()
     return args