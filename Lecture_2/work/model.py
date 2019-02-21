import torch.nn as nn

class AlexNet(nn.Module):
	def __init__(self, num_classes):
		super(AlexNet, self).__init__()

		self.num_classes = num_classes

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2)
		self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
		self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
		self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
		self.dropout = nn.Dropout()
		self.ff1 = nn.Linear(256*6*6, 4096)
		self.ff2 = nn.Linear(4096, 4096)
		self.output = nn.Linear(4096, num_classes)

	def forward(self, x):
		y = self.conv1(x)
		y = self.relu(y)
		y = self.maxpool(y)
		y = self.conv2(y)
		y = self.relu(y)
		y = self.maxpool(y)
		y = self.conv3(y)
		y = self.relu(y)
		y = self.conv4(y)
		y = self.relu(y)
		y = self.conv5(y)
		y = self.relu(y)
		y = self.maxpool(y)
		y = self.avgpool(y)
		y = y.view(y.size(0), 256*6*6)
		y = self.dropout(y)
		y = self.ff1(y)
		y = self.relu(y)
		y = self.dropout(y)
		y = self.ff2(y)
		y = self.relu(y)
		y = self.output(y)
		return y
