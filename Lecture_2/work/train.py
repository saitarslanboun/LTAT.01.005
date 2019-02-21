from DataIterator import *
from torch.utils.data import DataLoader
from model import *
from tqdm import tqdm

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def train_epoch(model, train_data_iterator, optimizer, epoch, opt):
	''' Epoch operation in training phase '''
	model.train()

	t = tqdm(train_data_iterator, mininterval=1, desc='-(Training)', leave=False)
	cntr = 0
	total_loss = 0
	for batch in t:
		inp, lbl = batch
		if torch.cuda.is_available():
			inp = inp.cuda()
			lbl = lbl.cuda()
		pred = model(inp)
		loss = F.mse_loss(pred, lbl)
		loss.backward()
		optimizer.step()
		description = "Loss: " + str(loss.item())
		t.set_description(description)
		cntr += 1
		total_loss += loss.item()
	avg_loss = total_loss / float(cntr)
	print "Training loss: " + str(avg_loss)
	fname = os.path.join(opt.save_dir, str(epoch) + ".chkpt")
	torch.save(model.state_dict(), fname)

def eval_epoch(model, valid_data_iterator, optimizer):
	''' Epoch operation in evaluation phase '''
	model.eval()

	t = tqdm(valid_data_iterator, mininterval=1, desc='-(Validation)', leave=False)
	total_loss = 0
	cntr = 0
	for batch in t:
		inp, lbl = batch
		if torch.cuda.is_available():
			inp = inp.cuda()
			lbl = lbl.cuda()
		pred = model(inp)
		loss = F.mse_loss(pred, lbl)
		total_loss += loss.item()
		cntr += 1
	avg_loss = total_loss / float(cntr)
	print "Validation loss: " + str(avg_loss)

def train(model, train_data_iterator, valid_data_iterator, optimizer, opt):
	''' Start training '''
	num_of_epochs = opt.num_of_epochs
	for epoch in range(num_of_epochs):
		train_epoch(model, train_data_iterator, optimizer, epoch, opt)
		eval_epoch(model, valid_data_iterator, optimizer)

if __name__ == '__main__':
	''' Main function '''
	parser = argparse.ArgumentParser()
	parser.add_argument("-train_inp", required=True)
	parser.add_argument("-train_lbl", required=True)
	parser.add_argument("-val_inp", required=True)
	parser.add_argument("-val_lbl", required=True)
	parser.add_argument("-image_dir", required=True)
	parser.add_argument("-batch_size", type=int, default=64)
	parser.add_argument("-num_of_epochs", type=int, default=100)
	parser.add_argument("-save_dir", required=True)
	opt = parser.parse_args()

	#Loading Dataset
	train_data_loader = LoadData(opt.image_dir, opt.train_inp, opt.train_lbl)
	train_data_iterator = DataLoader(train_data_loader, batch_size=opt.batch_size)
	valid_data_loader = LoadData(opt.image_dir, opt.val_inp, opt.val_lbl)
	valid_data_iterator = DataLoader(valid_data_loader, batch_size=opt.batch_size)
	num_classes = train_data_loader.class_num()
	opt.num_classes = num_classes

	model = AlexNet(num_classes)
	if torch.cuda.is_available():
		model.cuda()

	optimizer = torch.optim.Adam(model.parameters(), lr=0.0000001)
	train(model, train_data_iterator, valid_data_iterator, optimizer, opt)
