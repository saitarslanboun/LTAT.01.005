from torch.utils.data.dataset import Dataset

import os
import pretrainedmodels.utils as utils
import torch
import codecs

class LoadData(Dataset):
	def __init__(self, cnn_model, image_dir, index_file):
		self.load_img = utils.LoadImage()
		self.tf_img = utils.TransformImage(cnn_model)
		self.image_dir = image_dir
		self.image_indices = codecs.open(index_file, encoding="utf-8").readlines()

	def __getitem__(self, index):
		path_img = os.path.join(self.image_dir, self.image_indices[index].strip())
		img = self.load_img(path_img)
		img = self.tf_img(img)

		return img

	def __len__(self):
		return len(self.image_indices)
