class Attention(nn.Module):
	def __init__(self, hs):
		super(Attention, self).__init__()

		self.w_t = nn.Linear(hs, hs)
		self.w_v = nn.Linear(hs, hs)

		self.hs = hs

		self.softmax = nn.Softmax(dim=2)

	def forward(self, t, v):
		
		residual = t
		
		q = self.w_t(q)
		v = self.w_v(q)

		attn = self.softmax(torch.bmm(q, v.transpose(1, 2)) / sqrt(float(self.hs)))
		output = torch.bmm(attn, v) + residual
		
