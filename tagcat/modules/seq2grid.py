import random

import torch
import torch.nn as nn

# some default values to use
ROWS_FIXED_ADD = 0
COLS_FIXED_ADD = 2
ROWS_RAND_ADD = 3
COLS_RAND_ADD = 3


class GridSizeHandler:

	def __init__(self, min_grid_rows_add=0, min_grid_cols_add=2, range_grid_rows_add=3, range_grid_cols_add=3):
		self.min_grid_rows_add = min_grid_rows_add
		self.min_grid_cols_add = min_grid_cols_add
		self.range_grid_rows_add = range_grid_rows_add
		self.range_grid_cols_add = range_grid_cols_add


	def decide_grid_size(self, max_opsnum, max_opsize):
		rows = max_opsnum + self.min_grid_rows_add + random.randint(a=0, b=self.range_grid_rows_add)
		cols = max_opsize + self.min_grid_cols_add + random.randint(a=0, b=self.range_grid_cols_add)

		return rows, cols




NUM_ACTIONS = 3

class Seq2Grid(nn.Module):
	""" Seq2Grid module as defined in https://arxiv.org/abs/2101.04921, but the grid is mirrored to match the order from left to right.
	"""

	def __init__(self, embedding_size, num_units, device=None):
		"""
		Parameters
		----------
		embedding_size : int
				size of embedding vectors.
		num_units : int
				number of neurons for the hidden layers of the action decoder.
		device : None
				computing device to be used.
		"""
		super(Seq2Grid, self).__init__()

		self.device = device

		self.action_decoder = nn.Sequential(
			nn.Linear(in_features=embedding_size, out_features=num_units),
			nn.SiLU(),
			nn.Linear(in_features=num_units, out_features=NUM_ACTIONS),
			nn.Softmax(dim=-1)
		)

		self.num_units = num_units

	def forward(self, seq, num_lists=10, list_size=10):
		""" Produce the grid representation of a sequence of tokens by computing an action probability for each element.
		This forward method is more space/time-efficient than the naive approach of weighting the grids resulting from each action with their
		respective probability. The proof for correctness is stil missing, but the intuitive idea is that it is
		possible to reverse the process, and weight embedding vectors of sequence <seq> with weight matrices of shape (B,H,W)
		where each cell holds the probability of having an embedding vector in that specific cell. These super-position matrices
		can be computed starting from the last vector (its position is trivially computed from the last actions) and proceeding
		backwards. Finally, the output grid is the weighted sum of super-position matrices and embedding vectors.

		Parameters
		----------
		seq: Tensor
				tensor of shape (B, S, E) containing sequences of embedded tokens where:
						- B: batch size
						- S: sequences length
						- E: embedding size
		num_lists : int
				H dimension for the output grid. Default 10.
		list_size : int
				W dimension for the output grid. Default 10.

		Returns
		-------
		G : Tensor
				grid of shape (B,H,W,E), resulting from the preprocessing of the input sequences <seq> through the actions.
		"""

		batch_size, seq_len, emb_size = seq.size()  # B,S,E
		actions = self.action_decoder(seq)  # B,S,A
		# action order: (at, ap, an)

		# placeholder for grid size decision
		height, width = num_lists, list_size

		def devolve(T, a, rows_sums):
			# kernels are B,1,3 and B,1,1,3
			weight_other_cols = torch.stack(
				[torch.zeros_like(a[:, 2:3]), a[:, 2:3], a[:, 0:1]], dim=2).unsqueeze(1)
			# grid is 1,B,H,W, B is used as channel dimension, this allow to apply a different kernel
			# for each sample in the batch using grouped convolution
			other_cols = torch.conv2d(T.unsqueeze(
				0), weight_other_cols, padding=1, groups=batch_size)[0, :, 1:-1, :-1]
			last_col = T[:, :, -1]
			sums = torch.roll(rows_sums, shifts=1, dims=1)
			sums[:, 0] = 0.
			sums = a[:, 1:2] * sums
			last_col = sums + a[:, 2:3] * last_col
			rows_sums = (a[:, 0:1] + a[:, 2:3]) * rows_sums + sums
			last_col = last_col.unsqueeze(-1)
			V = torch.cat([other_cols, last_col], dim=2)
			return V, rows_sums

		def grid_update(x, a, T):
			return ((a[:, 0:1].unsqueeze(-1)+a[:, 1:2].unsqueeze(-1)) * T).unsqueeze(-1) * x.unsqueeze(1).unsqueeze(1)

		T = torch.zeros((batch_size, height, width), device=self.device)
		T[:, 0, -1] = 1.
		rows_sums = torch.zeros((batch_size, height), device=self.device)
		rows_sums[:, 0] = 1.
		G = torch.zeros((batch_size, height, width, emb_size), device=self.device)

		for i, (acts, token) in enumerate(zip(reversed(torch.unbind(actions, dim=1)), reversed(torch.unbind(seq, dim=1)))):
			# apply inplace sum
			G += grid_update(token, acts, T)
			if i == seq_len-1:
				break
			T, rows_sums = devolve(T, acts, rows_sums)

		return G
