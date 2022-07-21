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

from tagcat.train import train_net

def main():

	# set seeds for reproducibility
	random.seed(0)
	np.random.seed(0)
	torch.manual_seed(0)

	# get device
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	# dataset parameters
	operands_num = (1, 4)
	operands_size = (1, 10)
	val_operands_num = (5, 6)
	val_operands_size = (11, 15)
	num_groups = 2
	batch_size = 64 * num_groups
	batches_num = 10
	result_pad_symb = PAD_SYMB

	# ######### datasets definitions #########
	train_dataset = BatchedSumSeqDataset(
		operands_num=operands_num, operands_size=operands_size,
		batches_num=batches_num, batches_size=batch_size,
		num_groups=num_groups,
		result_pad_symb=result_pad_symb
	)
	val_dataset = BatchedSumSeqDataset(
		operands_num=val_operands_num, operands_size=val_operands_size,
		batches_num=1, batches_size=batch_size,
		result_pad_symb=result_pad_symb
	)
	grid_size_handler = GridSizeHandler()

	# ######### network parameters #########
	# general params
	embedding_size = 128
	groups = 1
	heads = 8
	dropout = 0.1
	norm_pos = 'post'
	vocab_size = len(DEC_ADDITION_ALPHA)
	use_pondernet = True
	use_highway = False
	use_reset = False
	use_posenc = True
	# seq2grid params
	num_units_s2g = 128
	# main transformer params
	num_units_step = 512
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

	########## loss parameters #########
	# Define the loss function
	class_weights = get_class_weightings_pad(
		num_classes=len(DEC_ADDITION_ALPHA), pad_class=train_dataset.get_pad_index(),
		pad_weight=0.1, device=device
	)
	loss_fn = SampleWeightedLoss(nn.CrossEntropyLoss(weight=class_weights))
	avg_steps = None
	reg_fn = HaltingProbRegularizer(reg_term=0.05)
	if reg_fn.vanilla:
		print(reg_fn.lambda_p)

	########## runtime parameters #########
	train_max_steps = 40
	val_max_steps = 50

	########## optimizer and scheduler parameters #########
	# optimizer parameters
	start_lr = 1e-3
	wd = 10.
	epochs = 510
	checkpoint_period = 30
	# scheduler parameters
	warm_restart_period = 30
	eta_min = 5e-5
	# gradient parameters
	grad_clipping = 10.

	no_wd_params = []
	wd_params = []
	no_wd = ['bias', 'embedding', 'rel_emb', 'q_emb', 'norm']

	for n, p in net.named_parameters():
		if any(s in n for s in no_wd):
			no_wd_params.append(p)
		else:
			wd_params.append(p)
	

	net_params =  [
		{'params': no_wd_params, 'weight_decay':0.0},
		{'params': wd_params, 'weight_decay':wd}
	]

	optimizer = optim.AdamW(net_params, lr=start_lr)
	scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=warm_restart_period, eta_min=eta_min)

	########## construct filename #########
	filename_params = {
		'pondernet': use_pondernet,
		'highway': use_highway,
		'reset': use_reset,
		'norm_pos': norm_pos
	}
	filename = craft_filename(**filename_params)
	print('used filename:', filename)


	########## construct filename #########

	loss_logs = {'train_loss':[], 'tr_acc':[], 'val_acc':[], 'epoch': 0}
	if use_pondernet:
		loss_logs['n_updates_train'] = {'min':[], 'max':[], 'mean':[], 'std':[]}
		loss_logs['n_updates_eval'] = {'min':[], 'max':[], 'mean':[], 'std':[]}

	checkpoint_handler = CheckpointHandler(
		base_filename=filename, path='trained_models/std_sasa/',
		checkpoint_period=checkpoint_period,
		network=net, optimizer=optimizer, scheduler=scheduler,
		logs=loss_logs, log_epoch=True, device=device
	)

	train_net(
		net, loss_fn=loss_fn, reg_fn=reg_fn,
		optimizer=optimizer, scheduler=scheduler,
		epochs=epochs, grid_size_handler=grid_size_handler,
		train_max_steps=train_max_steps, train_dataset=train_dataset,
		use_pondernet=use_pondernet, grad_clipping=grad_clipping,
		disable_scheduler=False,
		val_max_steps=val_max_steps, val_dataset=val_dataset,
		checkpoint_handler=checkpoint_handler,
		device=device, loss_logs=loss_logs, verbose=False
	)


if __name__ == '__main__':
	main()
