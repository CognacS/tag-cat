import torch
import torch.nn as nn

from .seq2grid import Seq2Grid
from .transformer import TransformerLikeConv, extract_top_list


################################################################################################################
############################################ SEQ-IN-GRID-OUT MODULE ############################################
################################################################################################################


class SeqInputGridOutput(nn.Module):
	""" Seq-input grid-output module encapsulating a Seq2Grid module, a recurrent transformer block, and an optional dynamic halting mechanism.
	"""

	def __init__(self, embedding_size=16, num_units_s2g=128, pondernet=None, kernel_size=3, groups=1, heads=8, num_units_step=128,
				 dropout=0.1, highway_gates=False, reset_gates=False, norm_pos='post', device=None):
		"""

		Parameters
		----------
		embedding_size : int
			size of vectors of the input grid, i.e. the size of single input embedding vectors. Must be divisible by <groups> and <heads>. Default 16.
		num_units_s2g : int
			number of neurons for the hidden layers of the Seq2Grid action decoder. Default 128.
		pondernet : Pondernet, optional
			PonderNet network which handles the dynamic halting mechanism. If None, computations stops after a statically fixed number of steps. Default None.
		kernel_size : int
			width and height of the attention window. Default 3.
		groups : int
			number of groups for the QKV linear projections. See the documentation for AttentionConv/AttentionConvFull for more details about this parameter.
		heads : int
			number of heads when computing attention. When heads==embedding_size, the preferred attention conv will be AttentionConvFull as it is more time-efficient.
			See the documentation for AttentionConv/AttentionConvFull for more details about this parameter.
		num_units_step : int
			number of neurons for the hidden layers of the step function.
		dropout : float
			dropout probability for the step function.
		highway_gates : bool
			if True, add an highway gating system [y = H(x)*x + (1-H(x))*F(x)] in place of the usual residual [y = x + F(x)]. Default False.
		reset_gates : bool
			if True, add a reset gating system which forgets part of the input before applying the blocks. Default False.
		norm_pos : str
			normalization position. Can be one of ['pre', 'post', 'none'], where 'pre' applies it to the input (before reset gating),
			'post' applies it to the output (after residuals as usual in transformers) and 'none' does not apply any normalization. Default 'post'.
		device : None
			computing device to be used.
		"""
		super(SeqInputGridOutput, self).__init__()

		self.embedding_size = embedding_size
		self.dropout = dropout
		self.device = device

		# build seq2grid layer
		self.seq2grid = Seq2Grid(embedding_size=embedding_size, num_units=num_units_s2g, device=device)

		# build main blocks
		self.block = TransformerLikeConv(
			embedding_size=embedding_size, kernel_size=kernel_size, groups=groups, heads=heads,
			num_units_step=num_units_step, dropout=dropout,
			highway_gates=highway_gates, reset_gates=reset_gates, norm_pos=norm_pos,
			device=device
		)

		# set pondernet
		self.pondernet = pondernet
		if self.pondernet is not None:
			self.pondernet.set_rec_layer(self.block)



	def forward(self, seq, num_lists, list_size, max_steps, fixed_steps=False, training_data=None):
		""" Produce the result of the task by encoding the input sequence into a grid, and compute on it the result

		Parameters
		----------
		seq: Tensor
			tensor of shape (B, S, E) containing sequences of embedded tokens where:
				- B: batch size
				- S: sequences length
				- E: embedding size
		num_lists : int
			H dimension for the grid.
		list_size : int
			W dimension for the grid.
		max_steps : int
			maximum number of computation steps before stopping (both in dynamic and non-dynamic halting).
		fixed_steps : bool
			flag to easily change from dynamic (False) to non-dynamic (True) when using a dynamic halting mechanism. Default False.
		training_data : dict
			dictionary containing training data to be registered when using pondernet (such as probabilities for the pondernet loss function).

		Returns
		-------
		grid : Tensor
			tensor of shape (B, H, W, E) where:
				- B: batch size
				- H: height of the grid / <num_lists>
				- W: width of the grid / <list_size>
				- E: embedding size
		n_updates : Tensor
			tensor of shape (B, 1) containing the number of computation steps for each sample in the batch. If using a fixed halting, this will
			be filled with <max_steps>. If using a dynamic halting mechanism, this tensor is filled with the real computation steps, i.e.:
				- if in training mode: the number of steps before having a cumulative halting probability higher than a threshold;
				- if in evaluation mode: the number of steps before sampling an halting event.
			Intuitively, the average number of steps in training mode will be higher than in evaluation mode.
		"""

		# compute grid as "channel last"
		grid = self.seq2grid(seq, num_lists, list_size)

		# compute transformer-like convolutions in a pondernet fashion
		if self.pondernet is not None and not fixed_steps:
			grid, n_updates = self.pondernet(grid, max_steps, training_data)
		else:
			for _ in range(max_steps):
				grid = self.block(grid)
			n_updates = torch.full(
				(grid.shape[0], 1), max_steps, dtype=torch.float, device=self.device)

		return grid, n_updates

################################################################################################################
############################################# ARCHITECTURE WRAPPER #############################################
################################################################################################################


class WrapperSIGO(nn.Module):
	""" Wrapper class for sequence-in grid-out module, adding an embedding matrix for the input and a decoder for the output.
	"""

	def __init__(self, seq_in_grid_out, vocab_size):
		"""
		Parameters
		----------
		seq_in_grid_out : SeqInputGridOutput
			body of the network, that is, the wrapped component, receiving as input the embedded sequence, and returning the output grid.
		vocab_size : int
			size of the vocabulary, that is, the number of predicted classes (to build the classification decoder).
		"""
		super(WrapperSIGO, self).__init__()

		emb_size = seq_in_grid_out.embedding_size
		dropout = seq_in_grid_out.dropout

		# prepare embedding for the vocabulary (token encoder)
		self.embedding = nn.Embedding(vocab_size, emb_size)

		# set body
		self.body = seq_in_grid_out

		# prepare token decoder
		self.decoder = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(emb_size, vocab_size)
		)

	def forward(self, seq, num_lists, list_size, max_steps, fixed_steps=False, training_data=None):
		""" Embed the sequence of symbols, compute on it, then returns the log probabilities of each symbol on the top row

		Parameters
		----------
		seq : Tensor
			sequences of index tokens of shape (B, S) where:
				- B: batch size
				- S: sequences length
		num_lists : int
			H dimension for the grid.
		list_size : int
			W dimension for the grid.
		max_steps : int
			maximum number of computation steps before stopping (both in dynamic and non-dynamic halting).
		fixed_steps : bool
			flag to easily change from dynamic (False) to non-dynamic (True) when using a dynamic halting mechanism. Default False.
		training_data : dict
			dictionary containing training data to be registered when using pondernet (such as probabilities for the pondernet loss function).

		Returns
		-------
		logits : Tensor
			output logits tensor of shape (B, W, V) where:
				- B: batch size
				- W: width of the grid
				- V: size of the vocabulary
		n_updates : Tensor
			tensor of shape (B, 1) containing the number of computation steps for each sample in the batch
		"""

		# encode input symbols
		src = self.embedding(seq)
		# process sequence
		tgt, n_updates = self.body(src, num_lists, list_size, max_steps, fixed_steps, training_data)

		#extract top list from the grid
		top_list = extract_top_list(tgt)

		# compute logits
		logits = self.decoder(top_list)

		return logits, n_updates