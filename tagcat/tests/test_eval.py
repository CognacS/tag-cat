import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tasks.math.alphabets import PAD_SYMB
from tasks.math.addition.generators import DEC_ADDITION_ALPHA, BatchedSumSeqDataset

from tagcat.modules.seq2grid import GridSizeHandler
from tagcat.modules.pondernet import PonderNetModule
from tagcat.modules.architecture import SeqInputGridOutput, WrapperSIGO

from tagcat.functions.losses import get_class_weightings_pad, SampleWeightedLoss, HaltingProbRegularizer
from tagcat.checkpoints import craft_filename, CheckpointHandler

from tagcat.evaluation import test_net

def main():
	# set seeds for reproducibility
	random.seed(0)
	np.random.seed(0)
	torch.manual_seed(0)

	# get device
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	# dataset parameters
	operands_num = (2, 2)
	operands_size = (1, 602)
	num_groups = 2
	batch_size = 64 * num_groups
	result_pad_symb = PAD_SYMB

	# ######### dataset definition #########
	test_dataset = BatchedSumSeqDataset(
		operands_num=operands_num, operands_size=operands_size,
		batches_num=1, batches_size=batch_size,
		result_pad_symb=result_pad_symb
	)
	grid_size_handler = GridSizeHandler()

	# ######### network parameters #########
	# general params
	embedding_size = 64
	groups = 8
	heads = 8
	dropout = 0.1
	norm_pos = 'post'
	vocab_size = len(DEC_ADDITION_ALPHA)
	use_pondernet = True
	use_highway = False
	use_reset = False
	use_posenc = True
	# seq2grid params
	num_units_s2g = 64
	# main transformer params
	num_units_step = 256
	kernel_size = 3
	# pondernet params
	context_len = 3

	pondernet = None
	if use_pondernet:
		pondernet = PonderNetModule(
			context_len=context_len, embedding_size=embedding_size,
			heads=heads, dropout=dropout,
			highway_gates=use_highway, reset_gates=use_reset,
			norm_pos=norm_pos, use_posenc=use_posenc,
			device=device
		)
	body = SeqInputGridOutput(
		embedding_size=embedding_size, num_units_s2g=num_units_s2g, pondernet=pondernet,
		kernel_size=kernel_size, groups=groups, heads=heads, num_units_step=num_units_step,
		dropout=dropout, highway_gates=use_highway, reset_gates=use_reset,
		norm_pos=norm_pos, device=device
	)
	net = WrapperSIGO(body, vocab_size=vocab_size).to(device)

	########## construct filename #########
	filename_params = {
		'pondernet': use_pondernet,
		'highway': use_highway,
		'reset': use_reset,
		'norm_pos': norm_pos
	}
	filename = craft_filename(**filename_params)
	print('used filename:', filename)

	# load stashed network and logs
	checkpoint_handler = CheckpointHandler(
		base_filename=filename, path='trained_models/std_sasa/',
		network=net, device=device
	)
	checkpoint_handler.load_structures()

	########## runtime parameters #########
	max_steps = 40

	accuracies = test_net(
		net, max_steps=max_steps, test_dataset=test_dataset,
		grid_size_handler=grid_size_handler,
		device=device, verbose=True
	)

	print(accuracies)


if __name__ == '__main__':
	main()