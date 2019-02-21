from torch.utils.data.dataset import Dataset
from PIL import Image

import codecs
import os
import torchvision.transforms as transforms
import numpy
import torch

class LoadData(Dataset):
	def __init__(self, image_dir, input_file, label_file):
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		self.transform = transforms.Compose([
			transforms.RandomSizedCrop(227),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize])
		self.image_dir = image_dir
		self.inputs = codecs.open(input_file, encoding="utf-8").readlines()
		self.labels = codecs.open(label_file, encoding="utf-8").readlines()

	def __getitem__(self, index):
		path_img = os.path.join(self.image_dir, self.inputs[index].strip())
		img = Image.open(path_img)
		img = self.transform(img)
		lbl = numpy.zeros(len(set(self.labels)))
		ind = int(self.labels[index].strip())
		lbl[ind] = 1
		lbl = torch.FloatTensor(lbl)

		return img, lbl

	def __len__(self):
		return len(self.inputs)

	def class_num(self):
		return len(set(self.labels))
