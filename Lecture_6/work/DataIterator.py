from torch.utils.data.dataset import Dataset

import codecs
import os
import pretrainedmodels.utils as utils
import torch

class LoadData(Dataset):
	def __init__(self, cnn_model, image_dir, index_file, caption_file, vocab):
		self.load_img = utils.LoadImage()
		self.tf_img = utils.TransformImage(cnn_model)
		self.image_dir = image_dir
		self.image_indices = codecs.open(index_file, encoding="utf-8").readlines()
		self.captions = codecs.open(caption_file, encoding="utf-8").readlines()
		self.vocab = vocab
		self.vocab_keys = vocab.keys()
		self.max_len = 50

	def __getitem__(self, index):
		path_img = os.path.join(self.image_dir, self.image_indices[index].strip())
		img = self.load_img(path_img)
		img = self.tf_img(img)

		text = self.captions[index].strip()
		tokens = text.split()
		caption = [self.vocab[token] if token in self.vocab_keys else self.vocab['<UNK>'] for token in tokens]
		caption = [self.vocab['<BOS>']] + caption + [self.vocab['<EOS>']]
		caption = caption[:self.max_len] + (self.max_len - len(caption))*[self.vocab['<PAD>']]
		caption = torch.LongTensor(caption)

		return img, caption

	def __len__(self):
		return len(self.captions)
