from TestDataIterator import *
from torch.utils.data import DataLoader
from model import *
from tqdm import tqdm
from torch.autograd import Variable

import argparse
import pretrainedmodels
import torch
import json

def test(test_data_iterator, model, args, cap_dict):

	# Reversing dictionary to generate tokens from given vocab IDs
	inv_cap_dict = {v: k for k, v in cap_dict.iteritems()}

	model.eval()

	# Writing generated descriptions
	target = open(args.output, "w")
	t = tqdm(test_data_iterator, mininterval=1, desc='-(Testing)', leave=False)
	for batch in t:
		images = batch
		captions = torch.zeros(images.size(0), 1).long()
		if torch.cuda.is_available():
			images = Variable(images.cuda())
			captions = Variable(captions.cuda())
		
		# Predicting each word sequentially, for 50 timesteps (with initial captions value) 
		for a in range(49):
			pred = model.generate(images, captions)
			captions = pred

		# Reversing from IDs to tokens
		captions = captions.tolist()
		for a in range(len(captions)):
			# clipping "<BOS>", "<EOS>" and "<PAD>"
			IDs = captions[a][1:]
			if 1 in IDs:
				IDs = IDs[:IDs.index(1)]
			tokens = [inv_cap_dict[val] for val in IDs]
			sentence = " ".join(tokens) + "\n"
			target.write(sentence.encode("utf-8"))
	target.close()

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--img_dir", required=True)
	parser.add_argument("--test_img_ind", required=True)
	parser.add_argument("--cap_vocab", required=True)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--chkpt", required=True)
	parser.add_argument("--output", default="captions.txt")
	args = parser.parse_args()

	# Load pretrained AlexNet
	cnn_model = pretrainedmodels.__dict__['alexnet'](num_classes=1000, pretrained='imagenet')
	if torch.cuda.is_available():
		cnn_model.cuda()

	# Loading Dataset
	cap_dict = json.load(open(args.cap_vocab))
	test_data_loader = LoadData(cnn_model, args.img_dir, args.test_img_ind)
	test_data_iterator = DataLoader(test_data_loader, batch_size=args.batch_size)

	# Initiating Captioning Model
	model = CaptionNet(cnn_model, len(cap_dict))
	if torch.cuda.is_available():
		model.cuda()

	# Update model with checkpoint weights
	model.load_state_dict(torch.load(args.chkpt))

	# Generate captions
	test(test_data_iterator, model, args, cap_dict)
