from math import sqrt

import torch.nn as nn
import torch

class ImageEncoder(nn.Module):
	def __init__(self, model):
		super(ImageEncoder, self).__init__()

		# AlexNet backend
		modules = list(model.children())[:-7] # delete the last fc layers.
		partial_alexnet = nn.Sequential(*modules)

		self.partial_alexnet = partial_alexnet
		self.alexnet = model

	def forward(self, images):

		# Extracting regional features from 5th convolutional layer
		V = self.partial_alexnet(images)

		# Flattening the output over last 2 dimensions to obtain vector of regional features
		V = V.view(V.size(0), V.size(1), -1).transpose(1, 2)

		# Obtain global features from output layer
		g = self.alexnet(images).unsqueeze(1)

		return V, g

class Attention(nn.Module):
        def __init__(self, hs):
                super(Attention, self).__init__()

                self.w_t = nn.Linear(in_features=hs, out_features=hs)
                self.w_v = nn.Linear(in_features=hs, out_features=hs)

                self.hs = hs

                self.softmax = nn.Softmax(dim=2)

        def forward(self, t, v):

                residual = t

                t = self.w_t(t)
                v = self.w_v(v)

                attn = self.softmax(torch.bmm(t, v.transpose(1, 2)) / sqrt(float(self.hs)))
                output = torch.bmm(attn, v) + residual

		return output

class Decoder(nn.Module):
	def __init__(self, vocab_size):
		super(Decoder, self).__init__()

		self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=128)
		self.RNN = nn.GRU(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
		self.attention = Attention(256)
		self.dense = nn.Linear(in_features=1000, out_features=128)

	def forward(self, V, g, captions):

		# Word Embedding Layer
		captions = self.embed(captions)

		# Dense layer to project 1000 dimensional global feature vector to 128 dimensional vector
		g = self.dense(g)

		# Concatenation Layer
		conc_captions = torch.cat((g, captions), dim=1)

		# RNN layer
		output, _ = self.RNN(conc_captions)

		# Attention Layer
		output = self.attention(output, V)

		return output

class CaptionNet(nn.Module):
	def __init__(self, cnn_model, vocab_size):
		super(CaptionNet, self).__init__()

		self.encoder = ImageEncoder(cnn_model)
		self.decoder = Decoder(vocab_size)
		self.output = nn.Linear(256, vocab_size)

	def forward(self, images, captions):

		# Encoding Images
		V, g = self.encoder(images)

		# Decoding Captions
		output = self.decoder(V, g, captions)

		# Exclude the first hidden state output of RNN
		#output = output[:, 1:, :]

		# Output Layer to project from hidden size to vocabulary size for prediction
		output = self.output(output)

		# Flattening the output over the first two dimensions for loss calculation
		foutput = output.contiguous().view(-1, output.size(2))

		return output, foutput

	def generate(self, images, captions):
		
		# Encoding Images
		V, g = self.encoder(images)

		if torch.cuda.is_available():
			captions = captions.cuda()

		# Decoding Captions
		output = self.decoder(V, g, captions)

		# Output Layer to project from hidden size to vocabulary size for prediction
		output = self.output(output)

		pred = output.max(2)[1]

		return pred
