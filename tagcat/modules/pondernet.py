import math
import torch
import torch.nn as nn
import torch.nn.init as init

from .transformer import extract_top_list, apply_block, LAYER_NORM_POSITION_MODES, LAYER_NORM_NONE
from .auxiliarydefs import *


def ALiBiPositionalEmbedding(batch_size, num_lists, list_size, tgt_size, attn_heads, device):
	def get_slopes(n):
		def get_slopes_power_of_2(n):
			start = (2**(-2**-(math.log2(n)-3)))
			ratio = start
			return [start*ratio**i for i in range(n)]

		if math.log2(n).is_integer():
			# In the paper, we only train models that have 2^a heads for some a. This function has
			return get_slopes_power_of_2(n)
		else:  # some good properties that only occur when the input is a power of 2. To maintain that even
			# when the number of heads is not a power of 2, we use this workaround.
			closest_power_of_2 = 2**math.floor(math.log2(n))
			return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

	maxpos = num_lists
	# produce the coefficeint for each head (heads)
	slopes = torch.Tensor(get_slopes(attn_heads)).to(device)
	alibi = -slopes.unsqueeze(1).unsqueeze(1) * torch.arange(maxpos, device=device).unsqueeze(
		0).unsqueeze(0).expand(attn_heads, -1, -1)  # produce series (heads, 1, maxpos)
	# repeat for the batch size (bsz * heads, 1, maxpos)
	alibi = alibi.repeat_tile(batch_size, dim=0)
	# repeat sections of size list_size (bsz * heads, 1, maxpos * list_size)
	alibi = alibi.repeat_interleave(list_size, dim=2)
	# expand dimension for the targets (bsz * heads, tgt_size, maxpos * list_size)
	alibi = alibi.expand(-1, tgt_size, -1)
	return alibi


def attn_in_function(context, alibi, src_seq):
	return {'query': context, 'key': src_seq, 'value': src_seq, 'attn_mask': alibi}


def attn_out_function(context):
	return context[0]


class PonderNetModule(nn.Module):

	def __init__(self, context_len, embedding_size, heads=8, num_units_step=64, num_units_lambda=128, dropout=0.1,
				 highway_gates=False, reset_gates=False, norm_pos='post',
				 use_posenc=True, threshold=0.05, device=None):
		"""

		Parameters
		----------
		context_len : int
			number of context vectors. Each context vector holds a piece of information to be used for halting.
		embedding_size : int
			size of context vectors. This must be equals to input vectors size.
		heads : int
			number of heads when computing self-attention. Default 4.
		num_units_step : int
			number of neurons for the hidden layers of the step function. Default 128.
		num_units_lambda : int
			number of neurons for the hidden layers of the lambda computing function. Default 128.
		dropout : float
			dropout probability for the step function.
		highway_gates : bool
			if True, add an highway gating system [y = H(x)*x + (1-H(x))*F(x)] in place of the usual residual [y = x + F(x)]. Default False.
		reset_gates : bool
			if True, add a reset gating system which forgets part of the input before applying the blocks. Default False.
		norm_pos : str
			normalization position. Can be one of ['pre', 'post', 'none'], where 'pre' applies it to the input (before reset gating),
			'post' applies it to the output (after residuals as usual in transformers) and 'none' does not apply any normalization. Default 'post'.
		use_posenc : bool
			if True, use AliBi positional encoding in the context transformer. If False, computational steps taken in the context transformer are
			permutation invariant. Default True.
		threshold : float
			threshold for deciding when to stop during training. In particular, stop when the cumulative halting distribution reaches 1-threshold. Default 0.05.
		device : None
			computing device to be used.
		"""
		
		super(PonderNetModule, self).__init__()

		self.rec_layer = None
		self.threshold = threshold
		self.heads = heads
		self.use_posenc = use_posenc
		self.device = device

		# generate the starting context as numbers uniformly distributed in [-1, 1)
		self.starting_context = nn.Parameter(torch.rand(1, context_len, embedding_size), requires_grad=True)
		init.normal_(self.starting_context, 0, 1)

		# information propagation from the grid to the general context
		self.cross_attn = nn.MultiheadAttention(
			embedding_size, heads, batch_first=True
		)
		self.step = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(embedding_size, num_units_step),
			nn.SiLU(),
			nn.Dropout(dropout),
			nn.Linear(num_units_step, embedding_size)
		)

		self.attn_block = {
			'name': 'pondernet_cross_attn',
			'function': self.cross_attn
		}
		self.step_block = {
			'name': 'pondernet_step',
			'function': self.step
		}

		assert norm_pos in LAYER_NORM_POSITION_MODES, f'Norm position {norm_pos} not implemented, must be one of {LAYER_NORM_POSITION_MODES}'
		self.norm_pos = norm_pos

		if norm_pos != LAYER_NORM_NONE:
			self.norm1 = nn.LayerNorm(embedding_size)
			self.norm2 = nn.LayerNorm(embedding_size)
			self.attn_block['norm'] = self.norm1
			self.step_block['norm'] = self.norm2

		if highway_gates:
			self.gate_attn = nn.Sequential(
				nn.Linear(embedding_size, embedding_size), nn.Sigmoid())
			self.gate_step = nn.Sequential(
				nn.Linear(embedding_size, embedding_size), nn.Sigmoid())
			self.attn_block['gate'] = self.gate_attn
			self.step_block['gate'] = self.gate_step

		if reset_gates:
			self.reset_attn = nn.Sequential(
				nn.Linear(embedding_size, embedding_size), nn.Sigmoid())
			self.reset_step = nn.Sequential(
				nn.Linear(embedding_size, embedding_size), nn.Sigmoid())
			self.attn_block['reset'] = self.reset_attn
			self.step_block['reset'] = self.reset_step

		self.params = {
			'norm_pos': norm_pos,
			'reset_gates': reset_gates,
			'highway_gates': highway_gates
		}

		# function to produce current halting probabilities given context
		self.lambdas = nn.Sequential(
			nn.Linear(context_len * embedding_size, num_units_lambda),
			nn.SiLU(),
			nn.Linear(num_units_lambda, 1),
			nn.Sigmoid()
		)


	def set_rec_layer(self, rec_layer):
		""" Set a recurrent layer to call at each computational step.

		Parameters
		----------
		rec_layer : nn.Module
			module to call at each computational step
		"""
		self.rec_layer = rec_layer


	def forward(self, grid, max_steps, training_data=None):
		"""
		Parameters
		----------
		grid : Tensor
			grid tensor of shape (B, H, W, E) where:
				- B: batch size
				- H: height of the grid
				- W: width of the grid
				- E: embedding size
		max_steps : int
			absolute maximum number of recurrent steps that can be done
		training_data : dict
			dictionary with keys:
				- 'decoder': output decoder function computing the output probabilities
				- 'target': target sequence for each sample
				- 'loss_fn': loss function of choice
				- 'loss_value': cumulated loss value throughout each computational step

		Returns
		-------
		grid : Tensor
			tensor of shape (B, H, W, E), same as the input.
		n_updates : Tensor
			tensor of shape (B, 1) containing the number of computation steps for each sample in the batch.
		"""

		assert self.rec_layer is not None, 'Recurrent layer is not set'

		batchsize, H, W, E = grid.shape

		# expand stating_context to batch_size
		context = self.starting_context.repeat_tile(batchsize, dim=0)

		# compute S source sequence length, size of the flattened grid
		flat_seqlen = H * W

		# probabilities of not having halted at step n-1 (prod_{j=1}^{n-1} (1-\lambda_j))
		# [B, 1]
		not_halted_prob = torch.ones(batchsize, 1, device=self.device)

		# ps
		# [B, 1]
		halting_prob = torch.zeros(batchsize, 1, device=self.device)

		# sum of ps
		# [B, 1]
		sum_of_ps = torch.zeros(batchsize, 1, device=self.device)

		# setup n updates
		# [B, 1]
		n_updates = torch.zeros(batchsize, 1, device=self.device)

		# setup indices of still running
		still_running_indices = torch.arange(
			end=batchsize, dtype=torch.long, device=self.device)

		# create alibi positional encoding
		if self.use_posenc:
			alibi = ALiBiPositionalEmbedding(
				batch_size=batchsize, num_lists=H, list_size=W, tgt_size=context.shape[1], attn_heads=self.heads, device=self.device)
		else:
			alibi = None

		steps = 0

		def compute_loss_for_dict(grid, probs, running_indices, training_data):
			decoder = training_data['decoder']
			target = training_data['target'][running_indices]
			loss_fn = training_data['loss_fn']
			prediction = decoder(extract_top_list(grid[running_indices]))
			loss = loss_fn(prediction, target, probs)
			training_data['loss_value'] = training_data['loss_value'] + loss
			training_data['running_indices'].append(running_indices)
			training_data['h_probs'].append(probs)

		def try_halting_and_loss(grid, context, still_running_indices):

			# ######## INFORMATION PROPAGATION ########
			# apply embeddings (positional is missing, replaced by ALiBi)
			# flatten H-W dimensions of grid
			idx = still_running_indices
			src_seq = grid[idx].flatten(start_dim=1, end_dim=2)

			# propagate grid information on context and compute step
			curr_bsz = len(idx)
			curr_context = context[idx]
			if alibi is not None:
				curr_alibi = alibi[:curr_bsz*self.heads]
			else:
				curr_alibi = None
			# apply attention block
			curr_context = apply_block(curr_context, self.attn_block, self.params,
									   in_function=attn_in_function, in_args={
										   'alibi': curr_alibi, 'src_seq': src_seq},
									   out_function=attn_out_function)
			# apply step block
			curr_context = apply_block(
				curr_context, self.step_block, self.params)

			# ######## HALTING PROBABILITIES ########
			# concatenate context tokens
			cat_context = curr_context.flatten(start_dim=1)

			# compute lambdas (halting probability conditioned on not having halted)
			cond_halting_prob = self.lambdas(cat_context)
			context[idx] = curr_context

			# ######## HALTING CONDTIONS ########
			if self.training:  # if in training mode
				# compute halted if over the threshold (the current timestep is N, final step)
				new_halted = (sum_of_ps[idx] > (
					1 - self.threshold)).bool().detach().squeeze(-1)

				# compute p_n halting probabilities
				halting_prob[idx] = cond_halting_prob * not_halted_prob[idx]
				# compute probabilities of not having halted up to now
				not_halted_prob[idx] = (
					1-cond_halting_prob) * not_halted_prob[idx]
				# compute the cumulative probability of having halted
				sum_of_ps[idx] = sum_of_ps[idx] + halting_prob[idx]

			else:  # if in evaluation mode
				new_halted = torch.bernoulli(
					cond_halting_prob.detach().squeeze(-1)).bool()

			new_halted = new_halted | (steps >= max_steps)

			# halt those which have not halted yet but halted this time
			halted_indices = still_running_indices[new_halted]
			still_running_indices = still_running_indices[torch.logical_not(
				new_halted)]

			if self.training:
				# account for residual probability for halted runs: p_N = 1 - sum of probs up to N-1
				halting_prob[halted_indices] = halting_prob[halted_indices] + \
					1. - sum_of_ps[halted_indices]
				if training_data is not None:
					# save training data
					compute_loss_for_dict(
						grid, halting_prob[idx], idx, training_data)

			return still_running_indices

		# ############### MAIN LOOP ############### #
		# try halting at current timestep
		still_running_indices = try_halting_and_loss(
			grid, context, still_running_indices)

		# while there are any still running and the max_steps is not reached
		while len(still_running_indices) > 0 and steps < max_steps:

			# update number of steps for those still running
			n_updates[still_running_indices] += 1
			steps += 1

			# compute transition function of running samples
			grid[still_running_indices] = self.rec_layer(
				grid[still_running_indices])

			# try halting at current timestep
			still_running_indices = try_halting_and_loss(
				grid, context, still_running_indices)

		# ######################################### #

		return grid, n_updates


def reconstruct_probabilities(list_running_indices, list_halting_probs):
	starting_probs = list_halting_probs.pop(0)
	list_running_indices.pop(0)
	list_final_probs = [starting_probs]
	for running_indices, halting_probs in zip(list_running_indices, list_halting_probs):
		new_probs = torch.zeros_like(starting_probs)
		new_probs[running_indices] = halting_probs
		list_final_probs.append(new_probs)

	return torch.cat(list_final_probs, dim=1)
