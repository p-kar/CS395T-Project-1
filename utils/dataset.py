import os
import pdb
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, RandomCrop, ToTensor, RandomHorizontalFlip, Scale

def readFile(fname):
	with open(fname, 'r') as fp:
		content = fp.readlines()

	content = [x.strip() for x in content]
	data = [x.split('\t') for x in content]
	return data

class YearBookDataset(Dataset):
	"""Yearbook Dataset"""

	def __init__(self, root_dir, split='train', start_date=1900, img_size=171, nclasses=120, target_type='regression'):
		"""
		Args:
			root_dir: Path to the root directory
			split: One of ['train' | 'valid' | 'test']
			start_data: Least date in the dataset
			img_size: Output size of the cropped image
		"""
		self.root_dir = os.path.join(root_dir, split)
		self.annot_file = os.path.join(root_dir, split + '.txt')
		self.split = split
		self.annotations = readFile(self.annot_file)
		self.start_date = start_date
		self.img_size = img_size
		self.nclasses = nclasses
		self.target_type = target_type

		if split in ['train', 'valid']:
			self.transform = transforms.Compose([RandomCrop(img_size), Scale(224), RandomHorizontalFlip(), ToTensor()])
		else:
			self.transform = transforms.Compose([CenterCrop(img_size), Scale(224), ToTensor()])

	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir, self.annotations[idx][0])
		img_label = None
		year_label = None
		if self.split != 'test':
			img_label = int(self.annotations[idx][1]) - self.start_date
			year_label = int(self.annotations[idx][1])

			if self.target_type == 'regression':
				img_label = torch.FloatTensor([img_label]) / self.nclasses
		
		img = Image.open(img_name)
		img = self.transform(img)

		return {'image': img, 'image_name': img_name, 'label': img_label, 'year': year_label}
