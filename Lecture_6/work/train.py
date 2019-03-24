from model import *
from DataIterator import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

import argparse
import pretrainedmodels
import torch
import json
import os

def valid_epoch(valid_data_iterator, criterion, model):
	model.eval()
	t = tqdm(valid_data_iterator, mininterval=1, desc='-(Validation)', leave=False)
	total_loss = 0
	cntr = 0
	for batch in t:
		images, captions = batch
		if torch.cuda.is_available():
			images = Variable(images.cuda())
			captions = Variable(captions.cuda())
		gold = captions.contiguous().view(-1)
		captions = captions[:, :-1]
		pred, fpred = model(images, captions)
		print pred.argmax(2)[-1]
		loss = criterion(fpred, gold)
		description = "Loss: " + str(loss.item())
		t.set_description(description)
		cntr += 1
		total_loss += loss.item()

	avg_loss = total_loss / float(cntr)

	return avg_loss

def train_epoch(train_data_iterator, criterion, model, optimizer):
	t = tqdm(train_data_iterator, mininterval=1, desc='-(Training)', leave=False)
	total_loss = 0
	cntr = 0
	for batch in t:
		images, captions = batch
		if torch.cuda.is_available():
			images = Variable(images.cuda())
			captions = Variable(captions.cuda())
		gold = captions.contiguous().view(-1)
		captions = captions[:, :-1]
		model.train()
		model.zero_grad()
		_, pred = model(images, captions)
		loss = criterion(pred, gold)
		loss.backward()
		optimizer.step()   
                description = "Loss: " + str(loss.item())
                t.set_description(description)
                cntr += 1
                total_loss += loss.item()

	avg_loss = total_loss / float(cntr)

	return avg_loss

def train(train_data_iterator, valid_data_iterator, model, criterion, optimizer, args):
	total_step = len(train_data_iterator)

	# Start Training
	for epoch in range(10):
		print "Training for epoch " + str(epoch) + "."
		avg_train_loss = train_epoch(train_data_iterator, criterion, model, optimizer)
		print "Training Loss: " + str(avg_train_loss)
		avg_valid_loss = valid_epoch(valid_data_iterator, criterion, model)
		print "Validation Loss: " + str(avg_valid_loss)

	# After 3 epochs we save the model state dictionary.
	chkpt_name = os.path.join(args.chkpt_dir, "model.chkpt") 
	torch.save(model.state_dict(), chkpt_name)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--img_dir", required=True)
	parser.add_argument("--train_img_ind", required=True)
	parser.add_argument("--train_cap", required=True)
	parser.add_argument("--cap_vocab", required=True)
	parser.add_argument("--val_img_ind", required=True)
	parser.add_argument("--val_cap", required=True)
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--chkpt_dir", default="checkpoints")
	args = parser.parse_args()

	# Load pretrained AlexNet
	cnn_model = pretrainedmodels.__dict__['alexnet'](num_classes=1000, pretrained='imagenet')
	if torch.cuda.is_available():
		cnn_model.cuda()

	# Loading Dataset
	cap_dict = json.load(open(args.cap_vocab))
        train_data_loader = LoadData(cnn_model, args.img_dir, args.train_img_ind, args.train_cap, cap_dict)
        train_data_iterator = DataLoader(train_data_loader, batch_size=args.batch_size)
        valid_data_loader = LoadData(cnn_model, args.img_dir, args.val_img_ind, args.val_cap, cap_dict)
        valid_data_iterator = DataLoader(valid_data_loader, batch_size=args.batch_size)

	# Initiating Captioning Model
	model = CaptionNet(cnn_model, len(cap_dict))
	if torch.cuda.is_available():
		model.cuda()

	# Listing parameters to be finetuned
	params = list(model.decoder.parameters())

	# Loss function
	criterion = nn.CrossEntropyLoss()
	if torch.cuda.is_available():
		criterion.cuda()

	# Optimizer
	optimizer = torch.optim.Adam(params)

	# Train model
	train(train_data_iterator, valid_data_iterator, model, criterion, optimizer, args)
